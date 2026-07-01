#!/bin/bash
#SBATCH --job-name=snp_analysis_small_nodes
#SBATCH --partition=standard
#SBATCH --constraint=small        # Fordert Small Nodes an
#SBATCH --nodes=8                 # Mehr Nodes für gleiche CPU-Leistung
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24        # 24 CPUs pro Small Node
#SBATCH --mem=120G                # 120GB RAM pro Small Node (von 128GB)
#SBATCH --time=8:00:00            # Etwas mehr Zeit für mehr Nodes
#SBATCH --output=snp_analysis_%j.out
#SBATCH --error=snp_analysis_%j.err

# ==========================================
# SNP Analysis Pipeline - 8 Small Nodes
# ==========================================
# Alternative Konfiguration mit verfügbaren Small Nodes
# 192 CPUs, 960GB RAM, 8 Stunden Laufzeit
# Optimiert für 150 CSV-Dateien mit großen Datensätzen

echo "=========================================="
echo "SNP Analysis Pipeline - 8 Small Nodes"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node List: $SLURM_JOB_NODELIST"
echo "CPUs per node: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Total CPUs: $((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))"
echo "Total Memory: $((SLURM_JOB_NUM_NODES * SLURM_MEM_PER_NODE))GB"
echo "Start time: $(date)"
echo "=========================================="

# Load required modules
module purge
module load anaconda3/5.2.0

# Set environment variables for performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK

# Performance optimizations
export MKL_DYNAMIC=FALSE
export MKL_INTERFACE_LAYER=LP64,GNU
export MKL_THREADING_LAYER=GNU

# Work directly in the current directory where the script is executed
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

# Check if required packages are already available
echo "Checking Python environment..."
python -c "import pandas, numpy, matplotlib, seaborn; print('All packages available')" 2>/dev/null || {
    echo "Some packages missing, trying to install..."
    # Try pip instead of conda (more reliable on HPC)
    pip install --user pandas numpy matplotlib seaborn scipy
}

# Set Python optimizations
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1

# Run the analysis with available resources
echo "Starting SNP analysis with available Small Nodes:"
echo "- 8 Small Nodes (Standard-Partition)"
echo "- 192 CPUs total (24 per node)"
echo "- 960GB RAM total (120GB per node)"
echo "- 8 hours runtime"
echo "- All optimizations enabled"

# Run single dataset analysis
python run_single_analysis_max.py

echo "Analysis completed at: $(date)"

# Results are in the working directory
echo "Results are in working directory: $WORK_DIR"
echo "Job completed successfully at: $(date)"
echo "=========================================="
echo "Performance Summary:"
echo "- Nodes used: $SLURM_JOB_NUM_NODES"
echo "- Total CPUs: $((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))"
echo "- Total Memory: $((SLURM_JOB_NUM_NODES * SLURM_MEM_PER_NODE))GB"
echo "- Runtime: $(date)"
echo "- Node type: Small Nodes (verfügbar)"
echo "=========================================="
