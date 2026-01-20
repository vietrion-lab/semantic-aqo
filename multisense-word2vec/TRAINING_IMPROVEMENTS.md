# Training Improvements Summary

## Changes Made to Address Loss Plateau Issue

### 1. **Learning Rate Scheduler (ExponentialLR)**
- **File**: [src/sensate/pipeline/training/training_pipeline.py](src/sensate/pipeline/training/training_pipeline.py)
- **Implementation**: Added ExponentialLR scheduler with golden ratio decay (`gamma = 0.61803398875`)
- **Benefits**: 
  - Learning rate decays exponentially each epoch
  - Helps escape local minima
  - Allows fine-tuning in later epochs

### 2. **Enhanced Training Monitoring**
- **File**: [src/sensate/pipeline/training/training_pipeline.py](src/sensate/pipeline/training/training_pipeline.py)
- **Features Added**:
  - Real-time loss display in tqdm progress bar
  - Current learning rate shown in progress bar
  - Loss component breakdown (w2v, distillation, orthogonality) every 10 batches
  - Learning rate updates logged after each epoch

### 3. **Improved Model Initialization**
- **File**: [src/sensate/model/sensate.py](src/sensate/model/sensate.py)
- **Changes**:
  - **Output Embeddings**: Changed from `randn * 0.01` to `Xavier/Glorot uniform` initialization
  - **Reason**: Better gradient flow from the start, prevents vanishing gradients
  
### 4. **Better Loss Weight Balancing**
- **File**: [src/sensate/model/sensate.py](src/sensate/model/sensate.py)
- **New Weights**:
  ```python
  alpha_w2v = 1.0        # Word2Vec loss
  alpha_distill = 0.5    # ↑ Increased from 0.3
  alpha_orth = 0.05      # ↓ Reduced from 0.1
  alpha_ent = 0.005      # ↓ Reduced from 0.01
  alpha_l2 = 0.0001      # ↓ Reduced from 0.001
  ```
- **Reasoning**: 
  - Stronger distillation helps utilize BERT knowledge better
  - Reduced regularization prevents over-constraining the model

### 5. **Label Smoothing**
- **File**: [src/sensate/model/sensate.py](src/sensate/model/sensate.py)
- **Implementation**: Added `label_smoothing=0.1` to cross-entropy loss
- **Benefits**:
  - Prevents overconfident predictions
  - Reduces risk of getting stuck in poor local minima
  - Improves generalization

### 6. **Increased Learning Rate**
- **File**: [src/config.yaml](src/config.yaml)
- **Change**: `0.006 → 0.01`
- **Reasoning**: 
  - Combined with exponential decay, starts with stronger updates
  - Helps escape the initial plateau faster

### 7. **Better Sense Embedding Initialization**
- **File**: [src/sensate/pipeline/training/initialization.py](src/sensate/pipeline/training/initialization.py)
- **Change**: Noise magnitude increased from `0.01` to `0.1`
- **Purpose**: Better differentiation between sense embeddings from the start

### 8. **Loss Component Tracking**
- **File**: [src/sensate/model/sensate.py](src/sensate/model/sensate.py)
- **Feature**: Model now stores `last_loss_components` for debugging
- **Usage**: Can monitor which loss component is causing issues

## Expected Improvements

1. **Faster Convergence**: Better initialization + higher initial LR should help escape plateau faster
2. **Better Final Performance**: Exponential LR decay allows fine-tuning
3. **More Stable Training**: Label smoothing + gradient clipping prevent instability
4. **Better Monitoring**: Real-time loss components help diagnose issues quickly

## Monitoring During Training

Now you'll see in the tqdm bar:
```
Epoch 1/20: 100%|████| 156/156 [00:10<00:00, loss=2.1234, lr=0.010000, w2v=1.234, dist=0.456, orth=0.012]
```

## Troubleshooting Loss Plateau

If loss still plateaus at 2.5:

1. **Check loss components** - See which component is stuck
2. **Monitor gradients** - Add gradient norm logging
3. **Try different initialization** - Experiment with different random seeds
4. **Reduce batch size** - Try batch_size=16 for more frequent updates
5. **Check data quality** - Ensure embeddings are properly normalized

## Next Steps

1. Run training and monitor the new metrics
2. If still plateauing, check which loss component is the issue
3. Consider adding gradient norm monitoring
4. Experiment with different learning rates (0.005-0.02 range)
