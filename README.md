# AnalysisObjectTransformer Model

This repository contains the implementation of the AnalysisObjectTransformer model, a deep learning architecture designed for event classification with reconstructed-object inputs. MultiHeadAttention is used to extract the correlation between jets (hadrons) in the final state. Achieves state-of-the-art performance on final states which can be summarized as jets accompanied by missing transverse energy.

## Model Overview

The AnalysisObjectTransformer model is structured to process jet-level features (energy, mass, area, btag score) in any order (permutation invariance) and event-level features (angle analysis of missing energy and leading jets) to classify signal from background processes to enhance the sensitivity to rare BSM signatures.

### Components

- **Embedding Layers**: Transform input data into a higher-dimensional space for subsequent processing.
- **Attention Blocks (AttBlock)**: Utilize multi-head attention to capture dependencies between different elements of the input data.
- **Class Blocks (ClassBlock)**: Extend attention mechanisms to incorporate class tokens, enabling the model to focus on class-relevant features. Implementation based on "Going deeper with transformers": https://arxiv.org/abs/2103.17239
- **MLP Head**: A sequence of fully connected layers that maps the output of the transformer blocks to the final prediction targets.

## Usage

```python
from particle_transformer import AnalysisObjectTransformer

model = AnalysisObjectTransformer(input_dim_obj=..., input_dim_event=..., embed_dims=..., linear_dims1=..., linear_dims2=..., mlp_hidden_1=..., mlp_hidden_2=..., num_heads=...)
