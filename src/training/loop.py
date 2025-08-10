import torch
from tqdm.auto import tqdm
import evaluate
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.models.mask2former import get_preprocessor
import numpy as np
from src.utils.palette import remap_labels, label2id
from src.utils.training_summary import log_training_summary
from PIL import Image
import time
import wandb  # Add this import at the top

try:
    import psutil
except ImportError:
    psutil = None

def train(model, optimizer, train_dataloader, valid_dataloader, id2label_remapped, device, accelerator, epochs=2, scheduler=None, log_freq=10):
    log_training_summary(model, optimizer, train_dataloader, id2label_remapped, device, accelerator, epochs, scheduler)
    
    best_val_loss = float('inf')
    best_mean_iou = 0.0
    best_epoch = 0
    global_step = 0
    
    # Initialize metrics tracking
    running_loss = 0.0
    num_samples = 0
    
    # Verify wandb tracker is available
    try:
        wandb_tracker = accelerator.get_tracker("wandb")
        if not wandb_tracker:
            accelerator.print("Warning: Wandb tracker not found")
    except Exception as e:
        accelerator.print(f"Warning: Could not access wandb tracker: {e}")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        accelerator.print(f"Epoch: {epoch + 1}/{epochs}")
        
        # Log learning rate at start of epoch
        if scheduler is not None:
            current_lr = optimizer.param_groups[0]['lr']
            accelerator.print(f"Current learning rate: {current_lr:.6f}")
            
            # Use accelerator.log instead of wandb.log
            accelerator.log({"learning_rate": current_lr, "epoch": epoch}, step=global_step)
        
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        train_step = 0
        
        for idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")):
            step_start_time = time.time()
            optimizer.zero_grad()
            
            outputs = model(
                pixel_values=batch["pixel_values"],
                mask_labels=batch["mask_labels"],
                class_labels=batch["class_labels"],
            )
            
            loss = outputs.loss
            # Check for invalid loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                accelerator.print(f"WARNING: Invalid loss detected at step {idx}: {loss}")
                continue
            accelerator.backward(loss)

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            train_step += 1
            global_step += 1
            
            # Frequent logging
            if idx % log_freq == 0:
                avg_loss = running_loss / num_samples
                step_time = time.time() - step_start_time
                
                accelerator.print(f"Step {idx}: Loss: {loss.item():.6f}, Avg Loss: {avg_loss:.6f}")
                
                # Use accelerator.log for all logging
                log_dict = {
                    "train/loss_step": loss.item(),
                    "train/loss_avg": avg_loss,
                    "train/step_time": step_time,
                    "train/samples_per_second": batch_size / step_time if step_time > 0 else 0,
                    "global_step": global_step,
                    "epoch": epoch
                }
                
                if scheduler is not None:
                    log_dict["learning_rate"] = optimizer.param_groups[0]['lr']
                
                # Log memory usage if available
                if torch.cuda.is_available():
                    log_dict.update({
                        "system/gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                        "system/gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                    })
                
                accelerator.log(log_dict, step=global_step)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        
        # Calculate epoch metrics - DEFINE VARIABLES BEFORE USING
        epoch_avg_loss = epoch_loss / epoch_samples
        epoch_time = time.time() - epoch_start_time
        
        # VALIDATION PHASE
        accelerator.print("Starting validation...")
        val_start_time = time.time()
        model.eval()
        val_losses = []
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_dataloader, desc="Validation")):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    mask_labels=batch["mask_labels"],
                    class_labels=batch["class_labels"],
                )
                
                val_losses.append(outputs.loss.detach())
                
                preprocessor = get_preprocessor(len(id2label_remapped))
                target_sizes = [(m.shape[0], m.shape[1]) for m in batch["original_segmentation_maps"]]
                processed_predictions = preprocessor.post_process_semantic_segmentation(
                    outputs, target_sizes=target_sizes
                )
                
                for pred, orig_mask in zip(processed_predictions, batch["original_segmentation_maps"]):
                    gt_remapped = remap_labels(np.array(orig_mask), label2id)
                    refs_np = gt_remapped.cpu().numpy().astype(np.int64)
                    pred_np = pred.cpu().numpy().astype(np.int64) if hasattr(pred, 'cpu') else np.array(pred).astype(np.int64)
                    
                    if refs_np.shape != pred_np.shape:
                        pred_pil = Image.fromarray(pred_np.astype(np.uint8))
                        pred_pil = pred_pil.resize((refs_np.shape[1], refs_np.shape[0]), Image.NEAREST)
                        pred_np = np.array(pred_pil).astype(np.int64)
                    
                    all_predictions.append(pred_np)
                    all_references.append(refs_np)
        
        # Define validation time - MOVED HERE
        val_time = time.time() - val_start_time
        
        # Gather validation losses
        val_losses_tensor = torch.stack(val_losses)
        gathered_val_losses = accelerator.gather(val_losses_tensor)
        avg_val_loss = gathered_val_losses.mean().item()
        
        # Prepare tensors for gathering
        if all_predictions:
            pred_tensors = [torch.from_numpy(p) for p in all_predictions]
            ref_tensors = [torch.from_numpy(r) for r in all_references]
            max_h = max(t.shape[0] for t in pred_tensors + ref_tensors) if pred_tensors else 1
            max_w = max(t.shape[1] for t in pred_tensors + ref_tensors) if pred_tensors else 1
            
            def pad_tensor(t, target_h, target_w):
                pad_h = target_h - t.shape[0]
                pad_w = target_w - t.shape[1]
                return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), value=0)
            
            pred_tensors_padded = [pad_tensor(t, max_h, max_w) for t in pred_tensors]
            ref_tensors_padded = [pad_tensor(t, max_h, max_w) for t in ref_tensors]
            
            pred_stack = torch.stack(pred_tensors_padded).to(accelerator.device)
            ref_stack = torch.stack(ref_tensors_padded).to(accelerator.device)
        else:
            pred_stack = torch.empty(0, 1, 1).to(accelerator.device)
            ref_stack = torch.empty(0, 1, 1).to(accelerator.device)
        
        # Gather predictions and references
        gathered_preds = accelerator.gather(pred_stack)
        gathered_refs = accelerator.gather(ref_stack)
        
        # Compute metrics and log (only main process)
        mean_iou = 0.0
        if accelerator.is_main_process:
            if gathered_preds.numel() > 0:
                gathered_preds_list = [p.cpu().numpy() for p in gathered_preds]
                gathered_refs_list = [r.cpu().numpy() for r in gathered_refs]
                
                try:
                    metric = evaluate.load("mean_iou")
                    metric.add_batch(
                        references=gathered_refs_list,
                        predictions=gathered_preds_list
                    )
                    metric_result = metric.compute(
                        num_labels=len(id2label_remapped),
                        ignore_index=0,
                        reduce_labels=False
                    )
                    mean_iou = metric_result.get("mean_iou", 0.0)
                    per_class_iou = metric_result.get("per_category_iou", [])
                    
                    accelerator.print(f"Mean IoU: {mean_iou:.6f}")
                    
                    # Log comprehensive metrics using accelerator.log
                    log_dict = {
                        "epoch": epoch,
                        "train/loss_epoch": epoch_avg_loss,
                        "train/epoch_time": epoch_time,
                        "val/loss": avg_val_loss,
                        "val/mean_iou": mean_iou,
                        "val/validation_time": val_time,
                        "best_mean_iou": max(best_mean_iou, mean_iou),
                        "global_step": global_step
                    }
                    
                    # Log per-class IoU
                    if len(per_class_iou) > 0:
                        for class_idx, class_iou in enumerate(per_class_iou):
                            if class_idx < len(id2label_remapped):
                                class_name = id2label_remapped[class_idx]
                                log_dict[f"val/iou_{class_name}"] = class_iou
                    
                    accelerator.log(log_dict, step=global_step)
                    
                    # Update best metrics
                    if mean_iou > best_mean_iou:
                        best_mean_iou = mean_iou
                        
                except Exception as e:
                    accelerator.print(f"Metric computation error: {e}")
                    accelerator.print("Mean IoU: Could not compute")
            else:
                accelerator.print("Mean IoU: No predictions to evaluate")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        
        accelerator.print(f"Validation Loss: {avg_val_loss:.6f}")
        
        # Save best model
        is_best = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            is_best = True
            
            if accelerator.is_main_process:
                # Get run info through accelerator tracker
                run_id = None
                run_name = None
                try:
                    wandb_tracker = accelerator.get_tracker("wandb")
                    if wandb_tracker and hasattr(wandb_tracker, 'tracker'):
                        run_id = wandb_tracker.tracker.id
                        run_name = wandb_tracker.tracker.name
                except Exception:
                    pass
                
                # Save model locally
                if run_id:
                    model_save_path = f"Output/best_model_{run_id}.pt"
                elif run_name:
                    model_save_path = f"Output/best_model_{run_name}.pt"
                else:
                    model_save_path = f"Output/best_model.pt"
                
                torch.save(accelerator.unwrap_model(model).state_dict(), model_save_path)
                accelerator.print(f"Model saved with validation loss: {best_val_loss:.6f} at {model_save_path}")
                
                # Save model artifact through accelerator's tracker
                try:
                    wandb_tracker = accelerator.get_tracker("wandb")
                    if wandb_tracker and hasattr(wandb_tracker, 'tracker'):
                        artifact = wandb.Artifact(
                            name="model-best",
                            type="model",
                            description=f"Best model with val_loss {best_val_loss:.6f} and mean_iou {mean_iou:.6f}"
                        )
                        artifact.add_file(model_save_path)
                        wandb_tracker.tracker.log_artifact(artifact)
                        
                        # Update summary metrics
                        wandb_tracker.tracker.summary.update({
                            "best_val_loss": best_val_loss,
                            "best_mean_iou": best_mean_iou,
                            "best_epoch": best_epoch
                        })
                except Exception as e:
                    accelerator.print(f"Could not save wandb artifact: {e}")
        
        # Log epoch summary using accelerator.log
        accelerator.log({
            "epoch_summary/is_best": is_best,
            "epoch_summary/epoch_duration": epoch_time,
            "epoch_summary/samples_processed": epoch_samples,
            "epoch_summary/steps_per_epoch": train_step
        }, step=global_step)
        
        # Log validation loss for every epoch (explicitly for wandb)
        accelerator.log({"val/loss": avg_val_loss, "epoch": epoch}, step=global_step)
