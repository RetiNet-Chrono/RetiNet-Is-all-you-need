"""
Standalone inference script for loading trained RetiNet models
"""
import torch
from model import RetiNetLanguageModel
from tokenizer import RetiNetTokenizer
from generate import ResponseGenerator
from config import DEVICE


def load_model(checkpoint_path: str):
    """
    Load model from checkpoint with proper weight unpacking.
    Handles 'weights_only' compatibility issues.
    """
    # Load checkpoint (use weights_only=False for compatibility)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    vocab_size = len(checkpoint['word2id'])
    embed_dim = checkpoint['hyperparameters']['embed_dim']
    num_heads = checkpoint['hyperparameters']['num_heads']
    windows = checkpoint['hyperparameters']['windows']
    
    # Initialize tokenizer with saved vocab
    tokenizer = RetiNetTokenizer.__new__(RetiNetTokenizer)
    tokenizer.word2id = checkpoint['word2id']
    tokenizer.id2word = {int(k): v for k, v in checkpoint['id2word'].items()}
    tokenizer.embed_matrix = checkpoint['embed_matrix']
    
    # Initialize model
    model = RetiNetLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        embed_matrix=tokenizer.get_embedding_matrix(),
        num_heads=num_heads,
        windows=windows
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer


def interactive_mode(checkpoint_path: str):
    """Run interactive chat with trained model."""
    print("Loading RetiNet model...")
    model, tokenizer = load_model(checkpoint_path)
    generator = ResponseGenerator(model, tokenizer, DEVICE)
    
    print("\nRetiNet Chat Interface")
    print("Commands: 'quit' to exit, 'temp=X' to set temperature")
    print("-" * 40)
    
    current_temp = 1.0
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower().startswith('temp='):
            try:
                current_temp = float(user_input.split('=')[1])
                generator.temperature = current_temp
                print(f"Temperature set to {current_temp}")
            except:
                print("Invalid temperature value")
            continue
        elif not user_input:
            continue
        
        response = generator.generate(user_input)
        print(f"Bot:  {response}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "./checkpoints/retinet_final.pth"
    
    interactive_mode(checkpoint_path)