"""
RetiNet Configuration
Zero-Projection Sparse Attention Language Model
"""
import torch
from typing import List, Optional

# Model Architecture
EMBED_DIM: int = 50
NUM_HEADS: int = 4
MAX_SEQ_LEN: int = 50
VOCAB_SIZE: int = 2000

# Multi-Head Window Configuration
# None means global attention, integer means local window size
ATTENTION_WINDOWS: List[Optional[int]] = [5, 10, 20, None]

# Training Hyperparameters
BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-3
EPOCHS: int = 100
GRAD_CLIP: float = 1.0
DROPOUT: float = 0.1
LR_DECAY_STEP: int = 30
LR_DECAY_GAMMA: float = 0.5

# Special Token IDs (will be assigned dynamically)
PAD_TOKEN: str = '<pad>'
UNK_TOKEN: str = '<unk>'
USER_TOKEN: str = '<user>'
BOT_TOKEN: str = '<bot>'
EOS_TOKEN: str = '<eos>'
REPEAT_TOKEN: str = '<repeat>'

# Paths
VEC_FILE: str = './data/wordvec50d.txt'
DATA_FILE: str = './data/datas.json'
CHECKPOINT_DIR: str = './checkpoints'

# Hardware
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random Seeds for Reproducibility
SEED: int = 42