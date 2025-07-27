#!/usr/bin/env python3
"""
Gloo Connectivity Diagnostic and Fix Script for Megatron-LM

This script helps diagnose and fix common Gloo backend connectivity issues
that cause "connectFullMesh failed" errors in distributed training.
"""

import os
import sys
import socket
import subprocess
import argparse
from typing import List, Optional


def get_network_interfaces():
    """Get available network interfaces."""
    try:
        result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            # Fallback for systems without 'ip' command
            result = subprocess.run(['ifconfig'], capture_output=True, text=True)
            return result.stdout if result.returncode == 0 else "Could not get network interfaces"
    except FileNotFoundError:
        return "Network interface commands not available"


def check_port_availability(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def find_available_port(start_port: int = 29500, end_port: int = 30000) -> Optional[int]:
    """Find an available port in the given range."""
    for port in range(start_port, end_port):
        if check_port_availability(port):
            return port
    return None


def test_connectivity(master_addr: str, master_port: int) -> bool:
    """Test TCP connectivity to master address and port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            result = s.connect_ex((master_addr, master_port))
            return result == 0
    except Exception:
        return False


def get_recommended_env_vars(rank: int, world_size: int, master_addr: str, master_port: int) -> dict:
    """Get recommended environment variables for fixing Gloo issues."""
    env_vars = {
        # Basic distributed training variables
        'MASTER_ADDR': master_addr,
        'MASTER_PORT': str(master_port),
        'RANK': str(rank),
        'LOCAL_RANK': str(rank),  # Assuming 1 GPU per node, adjust if different
        'WORLD_SIZE': str(world_size),
        
        # Gloo-specific network interface settings
        'GLOO_SOCKET_IFNAME': 'eth0',  # Most common interface, may need adjustment
        
        # Timeout settings
        'GLOO_DEVICE_TRANSPORT': 'TCP',
        
        # Alternative: Disable Gloo process groups if not needed
        # This can be set in training scripts
        'DISABLE_GLOO_PROCESS_GROUPS': 'false',  # Set to 'true' to disable
        
        # NCCL settings as backup
        'NCCL_SOCKET_IFNAME': 'eth0',
        'NCCL_IB_DISABLE': '1',  # Disable InfiniBand if not available
        'NCCL_DEBUG': 'INFO',    # For debugging
        
        # Timeout settings
        'NCCL_TIMEOUT': '600',   # 10 minutes timeout
    }
    
    return env_vars


def generate_training_script_fixes():
    """Generate code fixes for training scripts."""
    fixes = []
    
    # Fix 1: Disable Gloo process groups if not needed
    fix1 = '''
# Fix 1: Disable Gloo process groups in initialize_model_parallel call
# Add this parameter to your initialize_model_parallel call:

from megatron.core import parallel_state

parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=args.tensor_model_parallel_size,
    pipeline_model_parallel_size=args.pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
    create_gloo_process_groups=False,  # <-- Add this line
    distributed_timeout_minutes=30,    # <-- Increase timeout
)
'''
    fixes.append(("Disable Gloo Process Groups", fix1))
    
    # Fix 2: Network interface specification
    fix2 = '''
# Fix 2: Set network interface before initializing distributed training
# Add this before calling init_process_group or initialize_model_parallel:

import os

# Set network interface (adjust 'eth0' to your actual interface)
os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'  # or 'ens3', 'enp0s3', etc.
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

# For multi-interface systems, you can specify multiple interfaces:
# os.environ['GLOO_SOCKET_IFNAME'] = 'eth0,eth1,eth2,eth3'
'''
    fixes.append(("Network Interface Configuration", fix2))
    
    # Fix 3: Timeout and backend configuration
    fix3 = '''
# Fix 3: Configure timeouts and backend settings
# Modify your distributed initialization:

import torch.distributed as dist
from datetime import timedelta

# Increase timeout for slow networks
timeout = timedelta(minutes=30)  # Increase from default

# Initialize with explicit timeout
dist.init_process_group(
    backend='nccl',  # Use NCCL instead of mixed backends if possible
    timeout=timeout,
    init_method=f'tcp://{master_addr}:{master_port}',
    rank=rank,
    world_size=world_size
)
'''
    fixes.append(("Timeout and Backend Configuration", fix3))
    
    # Fix 4: Alternative initialization method
    fix4 = '''
# Fix 4: Alternative initialization with file-based method (for debugging)
# Use this if TCP-based initialization fails:

import tempfile
import torch.distributed as dist

# Create a shared file on a network filesystem
shared_file = "/shared/nfs/path/distributed_init_file"  # Adjust path
# Or use a temporary file for testing:
# shared_file = f"/tmp/distributed_init_{os.getpid()}"

dist.init_process_group(
    backend='nccl',
    init_method=f'file://{shared_file}',
    rank=rank,
    world_size=world_size,
    timeout=timedelta(minutes=30)
)
'''
    fixes.append(("File-based Initialization", fix4))
    
    return fixes


def diagnose_system():
    """Run system diagnostics."""
    print("=== GLOO CONNECTIVITY DIAGNOSTIC ===\n")
    
    print("1. Network Interfaces:")
    print(get_network_interfaces())
    print("\n" + "="*50 + "\n")
    
    print("2. Environment Variables Check:")
    important_vars = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 
                     'GLOO_SOCKET_IFNAME', 'NCCL_SOCKET_IFNAME']
    for var in important_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"   {var}: {value}")
    print("\n" + "="*50 + "\n")
    
    print("3. Port Availability Check:")
    master_port = int(os.environ.get('MASTER_PORT', '29500'))
    available = check_port_availability(master_port)
    print(f"   Port {master_port}: {'Available' if available else 'In Use'}")
    
    if not available:
        alt_port = find_available_port()
        if alt_port:
            print(f"   Alternative available port: {alt_port}")
    print("\n" + "="*50 + "\n")
    
    print("4. Connectivity Test:")
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    if master_addr and master_port:
        connected = test_connectivity(master_addr, master_port)
        print(f"   Connection to {master_addr}:{master_port}: {'SUCCESS' if connected else 'FAILED'}")
    print("\n" + "="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Diagnose and fix Gloo connectivity issues")
    parser.add_argument('--diagnose', action='store_true', help='Run system diagnostics')
    parser.add_argument('--generate-env', action='store_true', help='Generate environment variables')
    parser.add_argument('--generate-fixes', action='store_true', help='Generate code fixes')
    parser.add_argument('--rank', type=int, default=0, help='Process rank')
    parser.add_argument('--world-size', type=int, default=1, help='World size')
    parser.add_argument('--master-addr', default='localhost', help='Master address')
    parser.add_argument('--master-port', type=int, default=29500, help='Master port')
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_system()
    
    if args.generate_env:
        print("=== RECOMMENDED ENVIRONMENT VARIABLES ===\n")
        env_vars = get_recommended_env_vars(args.rank, args.world_size, 
                                          args.master_addr, args.master_port)
        
        print("# Export these environment variables before running training:")
        for key, value in env_vars.items():
            print(f"export {key}={value}")
        
        print("\n# Or create a script file:")
        print("cat > setup_distributed_env.sh << 'EOF'")
        for key, value in env_vars.items():
            print(f"export {key}={value}")
        print("EOF")
        print("source setup_distributed_env.sh")
        print("\n" + "="*50 + "\n")
    
    if args.generate_fixes:
        print("=== CODE FIXES FOR TRAINING SCRIPTS ===\n")
        fixes = generate_training_script_fixes()
        
        for i, (title, fix) in enumerate(fixes, 1):
            print(f"=== {title} ===")
            print(fix)
            print("\n" + "="*50 + "\n")
    
    if not any([args.diagnose, args.generate_env, args.generate_fixes]):
        print("No action specified. Use --help for options.")
        print("\nQuick start:")
        print("  python fix_gloo_connectivity.py --diagnose")
        print("  python fix_gloo_connectivity.py --generate-env --rank 0 --world-size 8")
        print("  python fix_gloo_connectivity.py --generate-fixes")


if __name__ == "__main__":
    main()