# Gloo Connectivity Fix for Megatron-LM

This repository contains fixes for the "Gloo connectFullMesh failed" error that occurs during distributed training with Megatron-LM.

## Error Description

The error typically appears as:
```
RuntimeError: Gloo connectFullMesh failed with [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:144] no error
```

This happens when processes cannot establish TCP connections to each other during Gloo process group initialization.

## Quick Fix (Recommended)

### 1. Apply Environment Variables Fix

```bash
# Source the fix script to set environment variables
source fix_gloo_issue.sh
```

### 2. Add Patch to Training Script

Add this line at the top of your training script (e.g., `moe_train_pipeline.py`):

```python
import gloo_patch  # This file is created by the fix script
```

### 3. Run Training

Your training should now work without Gloo connectivity issues.

## Detailed Solutions

### Solution 1: Disable Gloo Process Groups (Most Effective)

If you don't need Gloo process groups for your training, disable them:

```python
from megatron.core import parallel_state

parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=args.tensor_model_parallel_size,
    pipeline_model_parallel_size=args.pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
    create_gloo_process_groups=False,  # Add this line
    distributed_timeout_minutes=30,    # Increase timeout
)
```

### Solution 2: Network Interface Configuration

Set the correct network interface before training:

```bash
# Find your network interface
ip addr show

# Set environment variables (replace eth0 with your interface)
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_TIMEOUT=600
```

### Solution 3: Use NCCL Backend Only

Modify your distributed initialization to use NCCL only:

```python
import torch.distributed as dist
from datetime import timedelta

dist.init_process_group(
    backend='nccl',  # Use NCCL instead of mixed backends
    timeout=timedelta(minutes=30),
    # ... other parameters
)
```

### Solution 4: Alternative Initialization Method

If TCP-based initialization fails, use file-based method:

```python
import torch.distributed as dist
from datetime import timedelta

# Use a shared file on network filesystem
shared_file = "/shared/nfs/path/distributed_init_file"

dist.init_process_group(
    backend='nccl',
    init_method=f'file://{shared_file}',
    rank=rank,
    world_size=world_size,
    timeout=timedelta(minutes=30)
)
```

## Diagnostic Tools

### Run System Diagnostics

```bash
python fix_gloo_connectivity.py --diagnose
```

### Generate Environment Variables

```bash
python fix_gloo_connectivity.py --generate-env --rank 0 --world-size 8 --master-addr 192.168.1.100
```

### Generate Code Fixes

```bash
python fix_gloo_connectivity.py --generate-fixes
```

## Common Network Interface Names

Different systems use different network interface naming conventions:

- **Ubuntu/Debian**: `eth0`, `ens3`, `enp0s3`
- **CentOS/RHEL**: `eth0`, `ens4`, `enp0s8`
- **Cloud instances**: `ens3`, `ens4`, `eth0`
- **Docker**: `eth0`, `docker0`

To find your interface:
```bash
ip route | grep default | awk '{print $5}'
```

## Troubleshooting

### 1. Check Network Connectivity

```bash
# Test if nodes can reach each other
ping <master_node_ip>

# Test specific port connectivity
telnet <master_node_ip> <master_port>
# or
nc -zv <master_node_ip> <master_port>
```

### 2. Check Firewall Settings

```bash
# Check if firewall is blocking ports
sudo ufw status
sudo iptables -L

# Open necessary ports (adjust range as needed)
sudo ufw allow 29500:30000/tcp
```

### 3. Check Process Group Backend

Add debugging to see which backend is being used:

```python
import torch.distributed as dist

print(f"Backend: {dist.get_backend()}")
print(f"World size: {dist.get_world_size()}")
print(f"Rank: {dist.get_rank()}")
```

### 4. Enable Debug Output

```bash
export NCCL_DEBUG=INFO
export GLOO_DEBUG=1
```

## Environment Variables Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `GLOO_SOCKET_IFNAME` | Network interface for Gloo | `eth0` |
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL | `eth0` |
| `NCCL_IB_DISABLE` | Disable InfiniBand | `1` |
| `NCCL_TIMEOUT` | NCCL timeout in seconds | `600` |
| `MASTER_ADDR` | Master node address | `192.168.1.100` |
| `MASTER_PORT` | Master node port | `29500` |
| `WORLD_SIZE` | Total number of processes | `8` |
| `RANK` | Process rank | `0` |

## Files in This Fix

- `fix_gloo_issue.sh` - Main fix script that sets environment variables and creates patches
- `fix_gloo_connectivity.py` - Diagnostic tool for troubleshooting connectivity issues
- `gloo_patch.py` - Python patch module (created by fix script)
- `GLOO_FIX_README.md` - This documentation

## Usage Examples

### Single Node Training

```bash
# Source the fix
source fix_gloo_issue.sh

# Run training
python your_training_script.py --your-args
```

### Multi-Node Training

```bash
# On all nodes, source the fix
source fix_gloo_issue.sh

# Set node-specific environment variables
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=16
export RANK=0  # Different on each node

# Run training
python your_training_script.py --your-args
```

### With Slurm

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

# Source the fix on all nodes
source fix_gloo_issue.sh

# Run with srun
srun python your_training_script.py --your-args
```

## When to Use Each Solution

1. **Use Solution 1 (Disable Gloo)** if:
   - You don't need Gloo-specific features
   - You're using NCCL for GPU communication
   - You want the most reliable fix

2. **Use Solution 2 (Network Interface)** if:
   - You need Gloo process groups
   - You have multiple network interfaces
   - You're on a complex network setup

3. **Use Solution 3 (NCCL Only)** if:
   - You're doing GPU-only training
   - You want to avoid mixed backends
   - You have InfiniBand or high-speed interconnects

4. **Use Solution 4 (File-based)** if:
   - TCP initialization consistently fails
   - You have a shared filesystem
   - You're debugging connectivity issues

## Support

If these fixes don't resolve your issue:

1. Run the diagnostic tool: `python fix_gloo_connectivity.py --diagnose`
2. Check your network configuration and firewall settings
3. Verify that all nodes can reach each other on the specified ports
4. Consider using NCCL-only backend for GPU training

## Contributing

If you find additional fixes or improvements, please contribute them back to help others facing similar issues.