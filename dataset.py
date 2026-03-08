"""
Dialog Dataset with Causal Language Modeling Objective
"""
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List
from config import MAX_SEQ_LEN, DATA_FILE


class DialogDataset(Dataset):
    """Dataset for conversational AI training."""
    
    def __init__(
        self, 
        data_file: str, 
        tokenizer,
        max_len: int = MAX_SEQ_LEN,
        max_samples: int = None
    ):
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.data = raw_data[:max_samples] if max_samples else raw_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.pad_id = tokenizer.word2id['<pad>']
        self.user_id = tokenizer.word2id['<user>']
        self.bot_id = tokenizer.word2id['<bot>']
        self.eos_id = tokenizer.word2id['<eos>']
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        item = self.data[idx]
        
        # Tokenize context and response
        context_ids = self.tokenizer.encode(item['context'])
        response_ids = self.tokenizer.encode(item['response'])
        
        # Build full sequence: <user> context <bot> response <eos>
        input_ids = (
            [self.user_id] + context_ids + 
            [self.bot_id] + response_ids + 
            [self.eos_id]
        )
        
        # Truncate if necessary
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        
        # Create labels (shifted left by 1)
        labels = input_ids[1:] + [self.pad_id]
        
        # Mask context portion in loss computation (only train on response)
        # <user>(1) + context(len) + <bot>(1) positions are masked
        context_length = 1 + len(context_ids) + 1
        for i in range(min(context_length, len(labels))):
            labels[i] = -100  # CrossEntropyLoss ignore_index
        
        # Pad sequences
        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids = input_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )