#!/bin/bash

# Gloo Connectivity Fix Script for Megatron-LM
# This script provides immediate fixes for the "Gloo connectFullMesh failed" error

echo "=== GLOO CONNECTIVITY FIX SCRIPT ==="
echo "Applying fixes for 'Gloo connectFullMesh failed' error..."

# Function to detect network interface
detect_network_interface() {
    echo "Detecting network interface..."
    
    # Try common interface names
    for iface in eth0 ens3 enp0s3 ens4 enp0s8 bond0; do
        if ip link show "$iface" &>/dev/null; then
            echo "Found interface: $iface"
            export DETECTED_INTERFACE="$iface"
            return 0
        fi
    done
    
    # Fallback: get first non-loopback interface
    DETECTED_INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1)
    if [ -n "$DETECTED_INTERFACE" ]; then
        echo "Using default route interface: $DETECTED_INTERFACE"
        export DETECTED_INTERFACE
        return 0
    fi
    
    echo "Warning: Could not detect network interface, using eth0 as default"
    export DETECTED_INTERFACE="eth0"
}

# Function to set environment variables
set_environment_variables() {
    echo "Setting environment variables..."
    
    # Detect network interface
    detect_network_interface
    
    # Set Gloo network interface
    export GLOO_SOCKET_IFNAME="$DETECTED_INTERFACE"
    export NCCL_SOCKET_IFNAME="$DETECTED_INTERFACE"
    
    # Disable InfiniBand if not available
    export NCCL_IB_DISABLE=1
    
    # Set timeout values
    export NCCL_TIMEOUT=600
    export GLOO_DEVICE_TRANSPORT=TCP
    
    # Enable debug output
    export NCCL_DEBUG=INFO
    export GLOO_DEBUG=1
    
    echo "Environment variables set:"
    echo "  GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
    echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    echo "  NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
    echo "  NCCL_TIMEOUT=$NCCL_TIMEOUT"
}

# Function to check network connectivity
check_connectivity() {
    echo "Checking network connectivity..."
    
    if [ -n "$MASTER_ADDR" ] && [ -n "$MASTER_PORT" ]; then
        echo "Testing connection to $MASTER_ADDR:$MASTER_PORT..."
        if timeout 5 bash -c "</dev/tcp/$MASTER_ADDR/$MASTER_PORT"; then
            echo "✓ Connection successful"
        else
            echo "✗ Connection failed - this may cause Gloo issues"
            echo "  Make sure $MASTER_ADDR:$MASTER_PORT is accessible"
        fi
    else
        echo "MASTER_ADDR or MASTER_PORT not set, skipping connectivity test"
    fi
}

# Function to create a patched training script
create_patch_script() {
    cat > gloo_patch.py << 'EOF'
"""
Patch for Gloo connectivity issues in Megatron-LM training scripts.
Import this module at the beginning of your training script.
"""

import os
import sys
import socket
import time

def patch_gloo_initialization():
    """Apply patches to fix Gloo connectivity issues."""
    
    # Set network interface if not already set
    if 'GLOO_SOCKET_IFNAME' not in os.environ:
        # Try to detect the correct interface
        interfaces = ['eth0', 'ens3', 'enp0s3', 'ens4', 'enp0s8']
        for iface in interfaces:
            try:
                with open(f'/sys/class/net/{iface}/operstate', 'r') as f:
                    if f.read().strip() == 'up':
                        os.environ['GLOO_SOCKET_IFNAME'] = iface
                        os.environ['NCCL_SOCKET_IFNAME'] = iface
                        print(f"Auto-detected network interface: {iface}")
                        break
            except FileNotFoundError:
                continue
        else:
            # Fallback
            os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
            os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
            print("Using default network interface: eth0")
    
    # Set other important environment variables
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_TIMEOUT'] = '600'
    os.environ['GLOO_DEVICE_TRANSPORT'] = 'TCP'
    
    print("Gloo connectivity patch applied successfully")

def monkey_patch_initialize_model_parallel():
    """Monkey patch to disable Gloo process groups if needed."""
    try:
        from megatron.core import parallel_state
        original_init = parallel_state.initialize_model_parallel
        
        def patched_init(*args, **kwargs):
            # Add create_gloo_process_groups=False if not specified
            if 'create_gloo_process_groups' not in kwargs:
                kwargs['create_gloo_process_groups'] = False
                print("Disabled Gloo process groups to avoid connectivity issues")
            
            # Increase timeout
            if 'distributed_timeout_minutes' not in kwargs:
                kwargs['distributed_timeout_minutes'] = 30
                print("Increased distributed timeout to 30 minutes")
            
            return original_init(*args, **kwargs)
        
        parallel_state.initialize_model_parallel = patched_init
        print("Monkey patch for initialize_model_parallel applied")
        
    except ImportError:
        print("Could not import megatron.core.parallel_state - monkey patch skipped")

# Apply patches when module is imported
patch_gloo_initialization()
monkey_patch_initialize_model_parallel()

print("=" * 50)
print("GLOO CONNECTIVITY PATCHES APPLIED")
print("=" * 50)
EOF

    echo "Created gloo_patch.py - import this in your training script"
}

# Function to show usage instructions
show_usage() {
    echo ""
    echo "=== USAGE INSTRUCTIONS ==="
    echo ""
    echo "1. IMMEDIATE FIX - Run this script before training:"
    echo "   source fix_gloo_issue.sh"
    echo ""
    echo "2. TRAINING SCRIPT FIX - Add this to the top of your training script:"
    echo "   import gloo_patch  # This will be created by this script"
    echo ""
    echo "3. ALTERNATIVE - Disable Gloo process groups in your training script:"
    echo "   Add create_gloo_process_groups=False to initialize_model_parallel()"
    echo ""
    echo "4. NETWORK DEBUGGING - If issues persist:"
    echo "   python fix_gloo_connectivity.py --diagnose"
    echo ""
    echo "5. MANUAL ENVIRONMENT SETUP:"
    echo "   export GLOO_SOCKET_IFNAME=eth0  # Adjust interface name"
    echo "   export NCCL_SOCKET_IFNAME=eth0"
    echo "   export NCCL_IB_DISABLE=1"
    echo "   export NCCL_TIMEOUT=600"
    echo ""
}

# Main execution
main() {
    echo "Starting Gloo connectivity fix..."
    
    # Set environment variables
    set_environment_variables
    
    # Check connectivity if possible
    check_connectivity
    
    # Create patch script
    create_patch_script
    
    # Show usage instructions
    show_usage
    
    echo "Fix script completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Source this script: source fix_gloo_issue.sh"
    echo "2. Add 'import gloo_patch' to your training script"
    echo "3. Run your training with the applied fixes"
}

# Execute main function
main

# Export the function so it can be called from other scripts
export -f detect_network_interface
export -f set_environment_variables
export -f check_connectivity