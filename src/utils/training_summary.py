import torch
import inspect
import subprocess
import os
import socket
import sys

def log_training_summary(model, optimizer, train_dataloader, id2label_remapped, device, accelerator, epochs=2, scheduler=None):
    """
    Logs a detailed summary of the training configuration: model, optimizer, scheduler, transforms, batch size, processes, memory, CPU, and environment info.
    """
    # Model details
    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model
    model_name = getattr(base_model, 'name_or_path', str(type(base_model)))
    model_class = type(base_model).__name__
    try:
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        param_memory_bytes = total_params * 4  # float32
        param_memory_mb = param_memory_bytes / 1024**2
        # Estimate total training memory (params + gradients + optimizer states)
        est_training_memory_mb = param_memory_mb * 3  # rule of thumb
    except Exception:
        total_params = trainable_params = param_memory_mb = est_training_memory_mb = 'N/A'

    # Optimizer details
    optimizer_name = type(optimizer).__name__
    lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 'N/A'
    wd = optimizer.param_groups[0].get('weight_decay', 'N/A') if hasattr(optimizer, 'param_groups') else 'N/A'
    momentum = optimizer.param_groups[0].get('momentum', 'N/A') if hasattr(optimizer, 'param_groups') else 'N/A'
    # Only print key hyperparameters
    optimizer_summary = {'lr': lr, 'weight_decay': wd, 'momentum': momentum}
    optimizer_params = [dict((k, v) for k, v in group.items() if not callable(v) and not k.startswith('_')) for group in getattr(optimizer, 'param_groups', [])]

    # Scheduler details
    scheduler_name = type(scheduler).__name__ if scheduler is not None else 'None'
    scheduler_params = {}
    if scheduler is not None:
        for k in dir(scheduler):
            if not k.startswith('_') and not inspect.ismethod(getattr(scheduler, k)):
                try:
                    v = getattr(scheduler, k)
                    if isinstance(v, (int, float, str, bool, list, dict)):
                        scheduler_params[k] = v
                except Exception:
                    pass

    num_labels = len(id2label_remapped)
    # Try to get batch size robustly
    batch_size = None
    global_batch_size = None
    if hasattr(train_dataloader, 'batch_size') and train_dataloader.batch_size is not None:
        batch_size = train_dataloader.batch_size
    else:
        try:
            batch = next(iter(train_dataloader))
            if 'pixel_values' in batch:
                batch_size = batch['pixel_values'].shape[0]
        except Exception:
            batch_size = 'N/A'
    # Try to get global batch size
    try:
        num_processes = getattr(getattr(accelerator, 'state', None), 'num_processes', 1)
        global_batch_size = batch_size * num_processes if batch_size != 'N/A' and isinstance(batch_size, int) else 'N/A'
    except Exception:
        global_batch_size = 'N/A'

    # Transforms
    train_transform = getattr(getattr(train_dataloader, 'dataset', None), 'transform', None)
    target_transform = getattr(getattr(train_dataloader, 'dataset', None), 'target_transform', None)

    # Accelerate info
    num_processes = getattr(getattr(accelerator, 'state', None), 'num_processes', 'N/A')
    process_index = getattr(getattr(accelerator, 'state', None), 'process_index', 'N/A')
    local_process_index = getattr(getattr(accelerator, 'state', None), 'local_process_index', 'N/A')
    world_size = getattr(getattr(accelerator, 'state', None), 'num_processes', 'N/A')

    # Memory usage per GPU (allocated, reserved by this process)
    mem_str = "N/A"
    reserved_str = "N/A"
    if torch.cuda.is_available():
        try:
            mems = []
            reserved = []
            for i in range(torch.cuda.device_count()):
                mem_mb = torch.cuda.memory_allocated(i) / 1024**2
                res_mb = torch.cuda.memory_reserved(i) / 1024**2
                mems.append(f"GPU {i} (allocated): {mem_mb:.1f} MB")
                reserved.append(f"GPU {i} (reserved): {res_mb:.1f} MB")
            mem_str = ", ".join(mems)
            reserved_str = ", ".join(reserved)
        except Exception:
            mem_mb = torch.cuda.memory_allocated(device) / 1024**2
            res_mb = torch.cuda.memory_reserved(device) / 1024**2
            mem_str = f"{mem_mb:.1f} MB (allocated, GPU)"
            reserved_str = f"{res_mb:.1f} MB (reserved, GPU)"
    # Total GPU memory usage (all processes, via nvidia-smi)
    nvsmi_str = "N/A"
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        nvsmi_str = ", ".join([f"GPU {i}: {int(mb):,} MB (total used)" for i, mb in enumerate(lines)])
    except Exception:
        nvsmi_str = "(nvidia-smi not available)"

    # System RAM used by this process (detailed)
    try:
        import psutil
        process = psutil.Process()
        ram = process.memory_info()
        ram_str = f"{ram.rss/1024**2:.1f} MB (RSS), {ram.vms/1024**2:.1f} MB (VMS), {ram.shared/1024**2:.1f} MB (shared), {psutil.virtual_memory().percent}% used"
    except Exception:
        ram_str = "N/A"

    # CPU info
    try:
        import psutil
        cpu_str = f"Logical cores: {psutil.cpu_count(logical=True)}, Physical cores: {psutil.cpu_count(logical=False)}, Load: {psutil.getloadavg()}, CPU%: {psutil.cpu_percent()}"
    except Exception:
        cpu_str = "N/A"

    # Environment info
    env_info = {
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'),
        'Hostname': socket.gethostname(),
        'Python executable': sys.executable,
        'Conda env': os.environ.get('CONDA_DEFAULT_ENV', 'unset'),
        'Working dir': os.getcwd(),
        'User': os.environ.get('USER', 'unset'),
        'PATH': os.environ.get('PATH', '')
    }

    accelerator.print("\n===== Training Configuration Summary =====")
    accelerator.print(f"Model class: {model_class}")
    accelerator.print(f"Model name: {model_name}")
    accelerator.print(f"Num labels: {num_labels}")
    accelerator.print(f"Total parameters: {total_params}")
    accelerator.print(f"Trainable parameters: {trainable_params}")
    accelerator.print(f"Model parameter memory: {param_memory_mb if param_memory_mb != 'N/A' else 'N/A'} MB (float32)")
    accelerator.print(f"Estimated total training memory (params+grads+opt): {est_training_memory_mb if est_training_memory_mb != 'N/A' else 'N/A'} MB")
    accelerator.print(f"Device: {device}")
    accelerator.print(f"Optimizer: {optimizer_name} (summary: {optimizer_summary})")
    accelerator.print(f"Scheduler: {scheduler_name} {scheduler_params if scheduler_params else ''}")
    accelerator.print(f"Epochs: {epochs}")
    accelerator.print(f"Batch size (per process): {batch_size}")
    accelerator.print(f"Global batch size: {global_batch_size}")
    accelerator.print(f"Accelerate processes: {num_processes}")
    accelerator.print(f"Process rank: {process_index} (local: {local_process_index}) / world size: {world_size}")
    accelerator.print(f"GPU memory (allocated by this process): {mem_str}")
    accelerator.print(f"GPU memory (reserved by this process): {reserved_str}")
    accelerator.print(f"GPU memory (total, all processes): {nvsmi_str}")
    accelerator.print(f"System RAM (this process): {ram_str}")
    accelerator.print(f"CPU info: {cpu_str}")
    accelerator.print(f"Environment: {env_info}")
    accelerator.print(f"Train transform: {train_transform}")
    accelerator.print(f"Target transform: {target_transform}")
    accelerator.print("=========================================\n")
