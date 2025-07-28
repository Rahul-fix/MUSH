# Parallel and Distributed Training in Your Codebase (with Accelerate)

This document explains how parallel/distributed training is set up in your codebase using the HuggingFace Accelerate library, with specifics on batch handling, device placement, and loss calculation. Code snippets are included for clarity.

---

## 1. Distributed Setup with Accelerate

Your code uses the [Accelerate](https://huggingface.co/docs/accelerate/index) library to abstract away device placement and distributed training details. The key steps are:

```python
from accelerate import Accelerator
accelerator = Accelerator()
```

- **Device selection**: `accelerator.device` automatically chooses the correct device (CPU/GPU/TPU) for each process.
- **Distributed launch**: Your `train_distributed.sh` script launches the training with multiple GPUs using `accelerate launch`:

```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

---

## 2. DataLoader Preparation

You wrap your model and dataloaders with `accelerator.prepare`, which ensures that data and model are properly distributed across devices:

```python
model, train_dataloader, valid_dataloader = accelerator.prepare(model, train_dataloader, valid_dataloader)
```

- **Batch splitting**: Accelerate automatically splits each batch across available devices.
- **Batch size**: The batch size you set in the DataLoader is per process (per GPU). For example, if you set `batch_size=1` and use 3 GPUs, the effective batch size is 3 per step.

---

## 3. Batch and Tensor Dimensions

In your `segmentation_collate_fn`, the preprocessor returns tensors with shape:

- `pixel_values`: `(batch_size, C, H, W)`
- `mask_labels`: list of tensors, each `(num_masks, H, W)`
- `class_labels`: list of tensors, each `(num_masks,)`

Example:
```python
{
    "pixel_values": torch.Size([1, 3, 512, 512]),
    "mask_labels": [torch.Size([N, 512, 512])],
    "class_labels": [torch.Size([N])],
    "original_segmentation_maps": [...]
}
```

---

## 4. Model Forward and Loss Calculation

In your training loop (`src/training/loop.py`):

```python
outputs = model(
    pixel_values=batch["pixel_values"].to(device),
    mask_labels=[m.to(device) for m in batch["mask_labels"]],
    class_labels=[c.to(device) for c in batch["class_labels"]],
)
loss = outputs.loss
loss.backward()
```

- **Loss**: The model returns a single scalar loss per batch. Accelerate handles gradient synchronization across devices.
- **Optimizer step**: Each process computes gradients for its batch slice, then gradients are averaged (all-reduced) before the optimizer step.

---

## 5. Epochs, Steps, and Effective Batch Size

- **Epoch**: One pass over the entire dataset (split across all processes).
- **Step**: One optimizer update (each process works on its own batch slice).
- **Effective batch size**: `batch_size * num_processes` (e.g., 1 * 3 = 3).

---

## 6. Validation and Model Saving

- **Validation**: Each process evaluates its own validation batch slice. Metrics are typically gathered and averaged.
- **Model saving**: Only the main process saves the model (to avoid file corruption):

```python
if avg_val_loss < best_val_loss:
    torch.save(model.state_dict(), model_save_path)
```

---

## 7. Key Accelerate Features Used

- `Accelerator.prepare`: Wraps model and dataloaders for distributed training.
- `accelerator.device`: Handles device placement.
- `accelerate launch`: CLI for launching distributed jobs.

---

## 8. Further Reading

- [Accelerate Documentation](https://huggingface.co/docs/accelerate/index)
- [Distributed Training Concepts](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [HuggingFace Course: Distributed Training](https://huggingface.co/course/chapter7/6?fw=pt)

---

This should help you understand and research how Accelerate manages distributed training in your project.
