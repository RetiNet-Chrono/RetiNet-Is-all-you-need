"""
Training Pipeline for RetiNet
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import (
    DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, 
    GRAD_CLIP, CHECKPOINT_DIR, SEED, VEC_FILE, DATA_FILE,
    EMBED_DIM, NUM_HEADS, ATTENTION_WINDOWS, DROPOUT,
    LR_DECAY_STEP, LR_DECAY_GAMMA
)
from model import RetiNetLanguageModel
from tokenizer import RetiNetTokenizer
from dataset import DialogDataset
from generate import ResponseGenerator


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: nn.Module, 
    tokenizer: RetiNetTokenizer, 
    epoch: int, 
    loss: float,
    path: str = "retinet_checkpoint.pth"
):
    """Save model state and configuration."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'embed_matrix': tokenizer.embed_matrix,
        'word2id': tokenizer.word2id,
        'id2word': tokenizer.id2word,
        'hyperparameters': {
            'embed_dim': EMBED_DIM,
            'num_heads': NUM_HEADS,
            'windows': ATTENTION_WINDOWS,
        },
        'loss': loss,
    }, path)
    print(f"Checkpoint saved: {path}")


def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Execute one training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        logits = model(batch_x, padding_mask=(batch_x == 0))
        
        # Compute loss (flatten for CrossEntropy)
        loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def main():
    set_seed()
    
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize tokenizer and load data
    print("Loading tokenizer...")
    tokenizer = RetiNetTokenizer(VEC_FILE)
    vocab_size = len(tokenizer.word2id)
    
    print("Loading dataset...")
    dataset = DialogDataset(DATA_FILE, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True
    )
    print(f"Dataset size: {len(dataset)} samples")
    
    # Initialize model
    print("Initializing RetiNet model...")
    model = RetiNetLanguageModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        embed_matrix=tokenizer.get_embedding_matrix(),
        num_heads=NUM_HEADS,
        windows=ATTENTION_WINDOWS,
        dropout=DROPOUT,
        freeze_embed=False
    ).to(DEVICE)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=LR_DECAY_STEP, 
        gamma=LR_DECAY_GAMMA
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Initialize generator for periodic evaluation
    generator = ResponseGenerator(model, tokenizer, DEVICE)
    
    print("Starting training...")
    
    for epoch in range(EPOCHS):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, DEVICE)
        scheduler.step()
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} LR: {current_lr:.6f}")
        
        # Periodic evaluation every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("\n" + "="*40)
            print("Generation Test")
            print("="*40)
            test_cases = [
                "how are you",
                "what is your name", 
                "the kitchen stinks"
            ]
            for context in test_cases:
                response = generator.generate(context)
                print(f"User: {context}")
                print(f"Bot:  {response}")
            print("="*40 + "\n")
            
            # Save intermediate checkpoint
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, 
                f"retinet_epoch{epoch+1}.pth"
            )
            save_checkpoint(model, tokenizer, epoch+1, avg_loss, checkpoint_path)
    
    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "retinet_final.pth")
    save_checkpoint(model, tokenizer, EPOCHS, avg_loss, final_path)
    print("Training complete!")


if __name__ == '__main__':
    main()