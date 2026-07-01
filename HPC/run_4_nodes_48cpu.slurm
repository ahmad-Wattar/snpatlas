#!/bin/bash
#SBATCH --job-name=snp_analysis_4nodes_48cpu
#SBATCH --partition=standard
#SBATCH --nodelist=large[039-042]    # Spezifische Nodes mit 48 CPUs
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48           # 48 CPUs pro Node
#SBATCH --mem=256G                   # fast kompletter Node-RAM
#SBATCH --time=6:00:00
#SBATCH --output=snp_analysis_%j.out
#SBATCH --error=snp_analysis_%j.err

# ==========================================
# SNP Analysis Pipeline - 4 Large Nodes (48 CPUs)
# ==========================================
# Maximale Leistung: 4 spezifische Large Nodes mit 48 CPUs
# 192 CPUs, 1024GB RAM, 6 Stunden Laufzeit
# Optimiert für 150 CSV-Dateien mit großen Datensätzen

echo "=========================================="
echo "SNP Analysis Pipeline - 4 Large Nodes (48 CPUs)"
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
module load Python/3.9.6-GCCcore-11.2.0
module load matplotlib/3.5.1-foss-2021b
module load pandas/1.3.4-foss-2021b
module load seaborn/0.11.2-foss-2021b
module load numpy/1.21.2-foss-2021b

# Set environment variables for MAXIMUM performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK

# Performance optimizations
export MKL_DYNAMIC=FALSE
export MKL_INTERFACE_LAYER=LP64,GNU
export MKL_THREADING_LAYER=GNU

# Create working directory on fastest storage
WORK_DIR="/scratch/$USER/snp_analysis_$SLURM_JOB_ID"
mkdir -p $WORK_DIR
cd $WORK_DIR

# Copy data and scripts to scratch
echo "Copying data to scratch directory..."
cp -r /home/$USER/snpatlas/cursour\ 4\ HPC/* $WORK_DIR/

# Install required Python packages with optimizations
echo "Setting up optimized Python environment..."
pip install --user --upgrade pandas numpy matplotlib seaborn scipy

# Set Python optimizations
export PYTHONOPTIMIZE=2
export PYTHONDONTWRITEBYTECODE=1

# Run the analysis with maximum performance
echo "Starting SNP analysis with MAXIMUM performance:"
echo "- 4 Large Nodes (large[039-042])"
echo "- 192 CPUs total (48 per node)"
echo "- 1024GB RAM total (256GB per node)"
echo "- 6 hours runtime"
echo "- All optimizations enabled"

# Run single dataset analysis with MAXIMUM performance
python3 run_single_analysis_max.py

echo "Analysis completed at: $(date)"

# Copy results back to home directory
echo "Copying results back to home directory..."
RESULTS_DIR="/home/$USER/snp_results_4nodes_48cpu_$SLURM_JOB_ID"
mkdir -p $RESULTS_DIR
cp -r results/* $RESULTS_DIR/

echo "Results saved to: $RESULTS_DIR"
echo "Job completed successfully at: $(date)"
echo "=========================================="
echo "MAXIMUM Performance Summary:"
echo "- Nodes used: $SLURM_JOB_NUM_NODES"
echo "- Total CPUs: $((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK))"
echo "- Total Memory: $((SLURM_JOB_NUM_NODES * SLURM_MEM_PER_NODE))GB"
echo "- Runtime: $(date)"
echo "- Performance: MAXIMUM (48 CPUs per node)"
echo "=========================================="
