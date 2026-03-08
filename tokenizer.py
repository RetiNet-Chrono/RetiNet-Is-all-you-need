"""
Vocabulary and Embedding Management
Handles word vector loading and special token initialization
"""
import numpy as np
import torch
from typing import Dict, Tuple
from config import (
    EMBED_DIM, VOCAB_SIZE, PAD_TOKEN, UNK_TOKEN,
    USER_TOKEN, BOT_TOKEN, EOS_TOKEN, REPEAT_TOKEN
)


class RetiNetTokenizer:
    """Tokenizer with pretrained word vector support."""
    
    def __init__(self, vec_file: str, max_vocab: int = VOCAB_SIZE):
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.embed_matrix: np.ndarray = self._load_vectors(vec_file, max_vocab)
        
    def _load_vectors(self, file_path: str, max_vocab: int) -> np.ndarray:
        """Load word vectors and initialize special tokens."""
        word_vectors = {}
        
        # Load raw vectors
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < EMBED_DIM + 1:
                    continue
                word = parts[0]
                vector = list(map(float, parts[1:EMBED_DIM + 1]))
                word_vectors[word] = vector
                if len(word_vectors) >= max_vocab:
                    break
        
        print(f"Loaded {len(word_vectors)} word vectors from {file_path}")
        
        # Initialize vocabulary with special tokens
        vectors_list = []
        self.word2id[PAD_TOKEN] = 0
        self.id2word[0] = PAD_TOKEN
        vectors_list.append(np.zeros(EMBED_DIM))  # Pad token as zero vector
        
        self.word2id[UNK_TOKEN] = 1
        self.id2word[1] = UNK_TOKEN
        vectors_list.append(np.random.uniform(-0.1, 0.1, EMBED_DIM))
        
        next_id = 2
        
        # Filter out special tokens from raw vocabulary to avoid duplication
        reserved = {USER_TOKEN, BOT_TOKEN, EOS_TOKEN, REPEAT_TOKEN}
        
        for word in word_vectors:
            if word not in reserved and word not in self.word2id:
                self.word2id[word] = next_id
                self.id2word[next_id] = word
                vectors_list.append(word_vectors[word])
                next_id += 1
        
        # Initialize special tokens (use existing vector if available, else random)
        special_inits = {
            USER_TOKEN: word_vectors.get(USER_TOKEN, np.random.uniform(-0.1, 0.1, EMBED_DIM)),
            BOT_TOKEN: word_vectors.get(BOT_TOKEN, np.random.uniform(-0.1, 0.1, EMBED_DIM)),
            REPEAT_TOKEN: word_vectors.get(REPEAT_TOKEN, np.random.uniform(-0.1, 0.1, EMBED_DIM)),
            EOS_TOKEN: word_vectors.get(EOS_TOKEN, None)
        }
        
        # Fallback: EOS uses REPEAT vector if not found
        if special_inits[EOS_TOKEN] is None:
            special_inits[EOS_TOKEN] = special_inits[REPEAT_TOKEN].copy()
            print(f"Note: {EOS_TOKEN} not found, using {REPEAT_TOKEN} vector")
        
        for token in [USER_TOKEN, BOT_TOKEN, EOS_TOKEN, REPEAT_TOKEN]:
            if token not in self.word2id:
                self.word2id[token] = next_id
                self.id2word[next_id] = token
                vectors_list.append(special_inits[token])
                next_id += 1
        
        print(f"Vocab size: {len(self.word2id)}")
        print(f"Token IDs - User:{self.word2id[USER_TOKEN]}, "
              f"Bot:{self.word2id[BOT_TOKEN]}, EOS:{self.word2id[EOS_TOKEN]}")
        
        return np.stack(vectors_list, axis=0)
    
    def encode(self, text: str) -> list:
        """Convert text to token IDs."""
        return [self.word2id.get(token, self.word2id[UNK_TOKEN]) 
                for token in text.strip().split()]
    
    def decode(self, ids: list, skip_special: bool = True) -> str:
        """Convert token IDs to text."""
        special_ids = {self.word2id[PAD_TOKEN], self.word2id[EOS_TOKEN]}
        tokens = []
        for idx in ids:
            if idx in special_ids and skip_special:
                continue
            tokens.append(self.id2word.get(idx, UNK_TOKEN))
        return ' '.join(tokens)
    
    def get_embedding_matrix(self) -> torch.Tensor:
        """Return embedding matrix for model initialization."""
        return torch.from_numpy(self.embed_matrix).float()
    
    def save_vocab(self, path: str):
        """Save vocabulary for reproducibility."""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2id': self.word2id,
                'id2word': {str(k): v for k, v in self.id2word.items()},
                'embed_shape': self.embed_matrix.shape
            }, f, indent=2)