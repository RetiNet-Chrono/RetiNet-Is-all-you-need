"""
RetiNet Core Architecture
Naked Attention: Zero-Projection Sparse Attention Mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


class RetiNetLanguageModel(nn.Module):
    """
    RetiNet Language Model using multi-head naked attention.
    No Q/K projection matrices - operates directly on embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        embed_matrix: torch.Tensor,
        num_heads: int = 4,
        windows: Optional[List[Optional[int]]] = None,
        dropout: float = 0.1,
        freeze_embed: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Embedding layer initialized with pretrained vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embed_matrix).float())
        
        if freeze_embed:
            self.embedding.weight.requires_grad = False
            
        # Learnable per-head parameters for attention modulation
        # Self-attention reduction factor (diminishes diagonal attention)
        self.self_factors = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_heads)
        ])
        
        # Distance decay rate for temporal locality
        self.distance_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(num_heads)
        ])
        
        self.windows = windows if windows else [None] * num_heads
        
        # Output projections
        self.fusion = nn.Linear(num_heads * embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize non-embedding parameters."""
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass implementing naked attention mechanism.
        
        Args:
            input_ids: [batch_size, seq_len] token indices
            padding_mask: [batch_size, seq_len] bool mask (True for pad)
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed input tokens: [B, T, D]
        embeddings = self.embedding(input_ids)
        
        # Precompute positional distance matrix: [T, T]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        distance_matrix = torch.abs(positions - positions.T)
        
        # Causal mask (upper triangular) prevents attending to future tokens
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1
        )
        
        # Collect context vectors from each head
        head_contexts = []
        
        for head_idx in range(self.num_heads):
            alpha = self.distance_alphas[head_idx]
            self_factor = self.self_factors[head_idx]
            window = self.windows[head_idx]
            
            # Naked attention scores: raw dot product between embeddings
            # [B, T, D] @ [B, D, T] -> [B, T, T]
            attention_scores = torch.bmm(
                embeddings, 
                embeddings.transpose(1, 2)
            ) / math.sqrt(self.embed_dim)
            
            # Apply distance-based temporal decay: farther tokens contribute less
            # Decay = 1 / (1 + alpha * distance)
            decay_weights = 1.0 / (1.0 + alpha * distance_matrix.float())
            attention_scores = attention_scores * decay_weights.unsqueeze(0)
            
            # Reduce self-attention (diagonal) to avoid over-relying on current token
            diag_mask = torch.eye(seq_len, device=input_ids.device, dtype=torch.bool)
            attention_scores[:, diag_mask] *= self_factor
            
            # Apply local window mask if specified (sparse attention pattern)
            if window is not None:
                window_mask = (distance_matrix > window).unsqueeze(0).expand(batch_size, -1, -1)
                attention_scores.masked_fill_(window_mask, float('-inf'))
            
            # Enforce causal structure (autoregressive property)
            attention_scores.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
            
            # Mask padding positions (avoid attending to padded tokens)
            if padding_mask is not None:
                # Expand padding mask to [B, T, T] for key positions
                key_padding = padding_mask.unsqueeze(1).expand(batch_size, seq_len, -1)
                # Preserve diagonal to prevent -inf rows in softmax
                diag_3d = torch.eye(seq_len, device=input_ids.device).bool().unsqueeze(0)
                key_padding = key_padding & (~diag_3d)
                attention_scores.masked_fill_(key_padding, float('-inf'))
            
            # Normalize to probability distribution
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Aggregate values: [B, T, T] @ [B, T, D] -> [B, T, D]
            context = torch.bmm(attention_weights, embeddings)
            head_contexts.append(context)
        
        # Concatenate multi-head outputs and fuse
        multi_head_concat = torch.cat(head_contexts, dim=-1)  # [B, T, H*D]
        fused = self.fusion(multi_head_concat)  # [B, T, D]
        fused = self.dropout(fused)
        
        # Project to vocabulary logits
        logits = self.output_projection(fused)  # [B, T, V]
        
        return logits
    
    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)