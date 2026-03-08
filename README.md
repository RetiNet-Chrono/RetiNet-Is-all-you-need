# RetiNet: Naked Attention Language Model

RetiNet (Retina Network) is a lightweight language model using **zero-projection sparse attention** mechanism, designed for efficient deployment on edge devices.

## Architecture Highlights

- **Naked Attention**: No Q/K projection matrices, operates directly on embeddings
- **Multi-Head Sparse Attention**: Different attention windows per head [5, 10, 20, global]
- **Distance-Aware Decay**: Temporal locality through learnable distance decay
- **Parameter Efficient**: ~200K parameters for 50D embeddings, 4 heads

## Project Structure
