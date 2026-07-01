#!/bin/bash
#SBATCH --job-name=snp_analysis_simple
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=2                 # 2 Tasks auf demselben Node
#SBATCH --cpus-per-task=24         # je Task 24 physische Kerne
#SBATCH --mem=128G                 # insgesamt 124G, SLURM verteilt auf Tasks
#SBATCH --time=48:00:00
#SBATCH --output=snp_analysis_%j.out
#SBATCH --error=snp_analysis_%j.err
# ==========================================
# SNP Analysis Pipeline - Large Node Version
# ==========================================
echo "=========================================="
echo "SNP Analysis Pipeline - Large Node Version"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "CPUs per node: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Start time: $(date)"
echo "=========================================="

# Module laden
module purge
module load anaconda3/5.2.0

# Multi-Threading optimieren
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Arbeitsverzeichnis
WORK_DIR=$(pwd)
echo "Working directory: $WORK_DIR"

# Prüfen, ob Datensatz existiert
DATASET_DIR="dataset1"
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: $DATASET_DIR directory not found!"
    exit 1
else
    echo "Found dataset directory: $DATASET_DIR"
    ls -la "$DATASET_DIR" | head -10
fi

# Python-Umgebung prüfen
echo "Python version and path:"
python --version
python -c "import sys; print('Python executable:', sys.executable)"

# Pakete prüfen
echo "Checking required Python packages..."
python - <<END
try:
    import pandas as pd
    print("✓ pandas available")
except ImportError as e:
    print("✗ pandas error:", e)

try:
    import numpy as np
    print("✓ numpy available")
except ImportError as e:
    print("✗ numpy error:", e)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("✓ matplotlib available")
except ImportError as e:
    print("✗ matplotlib error:", e)

try:
    import seaborn as sns
    print("✓ seaborn available")
except ImportError as e:
    print("✗ seaborn error:", e)
END

# Analyse starten
echo "Starting SNP analysis..."
python run_single_analysis_simple.py

echo "=========================================="
echo "Analysis completed at: $(date)"
echo "Results directory: $WORK_DIR/results/"
echo "Job finished successfully"
echo "=========================================="

