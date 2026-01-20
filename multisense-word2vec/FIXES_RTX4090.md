# Fixes & RTX 4090 Optimizations

## 🔴 CRITICAL BUG FIX - Embedding ID Mapping

### Problem
**File**: `sensate/pipeline/training/training_pair.py`

**Bug**: Mismatch giữa embedding_id trong base_table và cách lookup trong dataset
- `base_table['embedding_id']` chứa index local của embedding trong embedding_table
- Code cũ lookup bằng `query_id` thay vì `embedding_id` → Sai hoàn toàn!

### Fix Applied
```python
# BEFORE (SAI):
embedding_dict = dict(zip(embedding_df['id'], embedding_df['embedding']))
bert_embeddings = self.embedding_dict[query_id]  # ❌ Lookup sai!

# AFTER (ĐÚNG):
embedding_ids_list = base_table['embedding_id'].values  # Extract embedding_id
self.embedding_ids = embedding_ids_list
bert_embeddings = self.embedding_dict[embedding_id]  # ✅ Lookup đúng!
```

### Impact
- **CRITICAL**: Bug này khiến model học sai embedding, mỗi training sample dùng sai BERT embedding
- **Ảnh hưởng**: Model performance sẽ rất kém vì distillation loss hoàn toàn sai
- **Status**: ✅ FIXED

---

## ⚡ RTX 4090 OPTIMIZATIONS

### Changes Summary

| Aspect | A100 (Old) | RTX 4090 (New) | Reason |
|--------|-----------|----------------|--------|
| **Mixed Precision** | BF16 | FP16 | RTX 4090 FP16 tensor cores nhanh hơn BF16 |
| **Batch Size Multiplier** | 4x | 2x | 24GB VRAM vs 40/80GB A100 |
| **Learning Rate** | 2x base | 1x base | Điều chỉnh theo batch size mới |
| **Tensor Core Mode** | BF16 reduction | FP16 reduction | Match với precision type |

### Detailed Changes

#### 1. Training Pipeline (`training_pipeline.py`)
```python
# Mixed precision: BF16 → FP16
acce = Accelerator(mixed_precision='fp16')  # Better for RTX 4090

# Batch size: 4x → 2x
batch_size=self.config.training.batch_size * 2  # 24GB VRAM

# Learning rate: không scale thêm
lr=self.config.training.learning_rate  # Removed 2x scaling
```

#### 2. BERT Extractor (`bert_extractor.py`)
```python
# Enable FP16 tensor cores
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

# Comment updated
print("RTX 4090 optimizations enabled")
```

### Expected Performance

**RTX 4090 specs:**
- Architecture: Ada Lovelace (TSMC 4N)
- CUDA Cores: 16,384
- Tensor Cores: Gen 4 (with FP8 support)
- VRAM: 24GB GDDR6X
- TDP: 450W
- FP16 Performance: ~83 TFLOPS
- Memory Bandwidth: 1,008 GB/s

**Performance vs A100:**
- Gaming/Consumer workloads: RTX 4090 faster (~1.5x)
- Training with FP16: ~Similar or slightly better
- Training with BF16: A100 better (dedicated BF16 cores)
- VRAM: A100 wins (40/80GB vs 24GB)

### Recommendations

1. **Batch Size**: Với 24GB VRAM, nếu OOM xảy ra:
   ```yaml
   # config.yaml
   training:
     batch_size: 4  # Giảm từ 4 xuống 2 hoặc 1
   ```

2. **Gradient Accumulation**: Nếu cần batch size lớn hơn:
   ```python
   # Trong training loop
   accumulation_steps = 2
   if (batch_idx + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

3. **Monitor VRAM**: 
   ```python
   torch.cuda.memory_summary()  # Check usage
   ```

---

## 📋 Testing Checklist

- [x] Fix embedding_id mapping bug
- [x] Change mixed precision to FP16
- [x] Adjust batch size multiplier to 2x
- [x] Adjust learning rate (remove 2x scaling)
- [x] Update tensor core settings for FP16
- [ ] Run training and verify no errors
- [ ] Monitor GPU utilization (`nvidia-smi`)
- [ ] Check training metrics improve vs before
- [ ] Verify model convergence

---

## 🚀 Next Steps (TODO)

### High Priority
1. **Add error handling** cho GPU OOM
2. **Extract magic numbers** to config (loss weights, etc.)
3. **Add early stopping** mechanism
4. **Add learning rate scheduler**

### Medium Priority
5. Refactor Trainer class (too large)
6. Add unit tests for data pipeline
7. Replace regex SQL parsing với proper parser
8. Add validation set support

### Low Priority
9. Add TensorBoard logging
10. Add model checkpointing with top-k
11. Add gradient clipping config
12. Add mixed precision config option

---

## 📝 Notes

- **[:5] limit**: Intentionally kept for testing - remove when ready for full training
- **torch.compile**: Có thể thêm compiler cache để tăng tốc khởi động lần 2+
- **FP8 support**: RTX 4090 có FP8 tensor cores, có thể thử trong tương lai với Transformer Engine
