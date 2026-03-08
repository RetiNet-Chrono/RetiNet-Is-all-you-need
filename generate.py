"""
Inference Interface for RetiNet
Greedy decoding and response generation
"""
import torch
import torch.nn.functional as F
from config import MAX_SEQ_LEN, DEVICE


class ResponseGenerator:
    """Text generation interface for RetiNet model."""
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        device: torch.device = DEVICE,
        temperature: float = 1.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.max_len = MAX_SEQ_LEN
        
        # Cache special token IDs
        self.user_id = tokenizer.word2id['<user>']
        self.bot_id = tokenizer.word2id['<bot>']
        self.eos_id = tokenizer.word2id['<eos>']
        self.pad_id = tokenizer.word2id['<pad>']
        self.unk_id = tokenizer.word2id['<unk>']
    
    def generate(self, context: str, max_response_len: int = 20) -> str:
        """
        Generate response for given context using greedy decoding.
        
        Args:
            context: Input context text
            max_response_len: Maximum response length
            
        Returns:
            Generated response string
        """
        self.model.eval()
        
        # Build input: <user> context <bot>
        context_ids = self.tokenizer.encode(context)
        input_ids = [self.user_id] + context_ids + [self.bot_id]
        
        generated_ids = []
        
        with torch.no_grad():
            for _ in range(max_response_len):
                input_tensor = torch.tensor([input_ids], device=self.device)
                
                # Forward pass
                logits = self.model(input_tensor, padding_mask=(input_tensor == 0))
                
                # Get next token prediction (last position)
                next_token_logits = logits[0, -1, :] / self.temperature
                next_token = torch.argmax(next_token_logits).item()
                
                # Stop conditions
                if next_token == self.eos_id or next_token == self.pad_id:
                    break
                    
                generated_ids.append(next_token)
                input_ids.append(next_token)
                
                # Hard length limit
                if len(input_ids) > self.max_len:
                    break
        
        return self.tokenizer.decode(generated_ids, skip_special=True)
    
    def generate_diverse(
        self, 
        context: str, 
        num_samples: int = 3,
        top_k: int = 10
    ) -> list:
        """
        Generate multiple diverse responses using top-k sampling.
        
        Args:
            context: Input context
            num_samples: Number of response variants
            top_k: Top-k sampling parameter
            
        Returns:
            List of response strings
        """
        self.model.eval()
        responses = []
        
        for _ in range(num_samples):
            context_ids = self.tokenizer.encode(context)
            input_ids = [self.user_id] + context_ids + [self.bot_id]
            generated_ids = []
            
            with torch.no_grad():
                for _ in range(20):
                    input_tensor = torch.tensor([input_ids], device=self.device)
                    logits = self.model(input_tensor)
                    next_logits = logits[0, -1, :]
                    
                    # Top-k filtering
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(next_logits / self.temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    if next_token in [self.eos_id, self.pad_id]:
                        break
                        
                    generated_ids.append(next_token)
                    input_ids.append(next_token)
                    
                    if len(input_ids) > self.max_len:
                        break
            
            responses.append(self.tokenizer.decode(generated_ids, skip_special=True))
        
        return responses