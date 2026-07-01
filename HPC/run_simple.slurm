#!/bin/bash
#SBATCH --job-name=snp_analysis_simple
#SBATCH --partition=standard
#SBATCH --constraint=small
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=snp_analysis_%j.out
#SBATCH --error=snp_analysis_%j.err

# ==========================================
# SNP Analysis Pipeline - Simple Version
# ==========================================
# Verwendet die bereits installierten Pakete
# 192 CPUs, 960GB RAM, 8 Stunden Laufzeit

echo "=========================================="
echo "SNP Analysis Pipeline - Simple Version"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "CPUs per node: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Total CPUs: $((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))"
echo "Total Memory: $((SLURM_JOB_NUM_NODES * SLURM_MEM_PER_NODE))GB"
echo "Start time: $(date)"
echo "=========================================="

# Load only anaconda3 module
module purge
module load anaconda3/5.2.0

# Set environment variables for performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Work directly in the current directory
WORK_DIR=$(pwd)
echo "Working directory: $WORK_DIR"

# Verify that dataset1 directory exists
echo "Checking for dataset1 directory..."
if [ -d "dataset1" ]; then
    echo "Found dataset1 directory with CSV files"
    ls -la dataset1/ | head -10
else
    echo "ERROR: dataset1 directory not found!"
    echo "Please make sure the dataset1 directory exists in the current directory"
    exit 1
fi

# Check Python and packages
echo "Checking Python environment..."
python --version
python -c "import sys; print('Python path:', sys.executable)"

# Try to import packages
echo "Testing package imports..."
python -c "
try:
    import pandas as pd
    print('✓ pandas available')
except ImportError as e:
    print('✗ pandas error:', e)

try:
    import numpy as np
    print('✓ numpy available')
except ImportError as e:
    print('✗ numpy error:', e)

try:
    import matplotlib
    matplotlib.use('Agg')  # Set backend before importing pyplot
    import matplotlib.pyplot as plt
    print('✓ matplotlib available')
except ImportError as e:
    print('✗ matplotlib error:', e)

try:
    import seaborn as sns
    print('✓ seaborn available')
except ImportError as e:
    print('✗ seaborn error:', e)
"

# Run the analysis
echo "Starting SNP analysis..."
echo "Working with available packages..."

# Use the simple robust version
python run_single_analysis_simple.py

echo "Analysis completed at: $(date)"
echo "Results are in: $WORK_DIR/results/"
echo "Job completed successfully at: $(date)"
