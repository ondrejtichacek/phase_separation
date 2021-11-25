#!/bin/bash
#SBATCH -J lattice
#SBATCH --qos=backfill
#SBATCH --partition=cpu,scpu,bfill
#SBATCH --time=0-04:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2400M
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ondrej.tichacek@uochb.cas.cz
#SBATCH --array=1-121%72

# $SLURM_ARRAY_JOB_ID = e.g. 1150533
# $SLURM_ARRAY_TASK_ID ... will change from 1 to 1225
# $SLURM_ARRAY_TASK_COUNT = 1225
# $SLURM_ARRAY_TASK_MAX = 1225
# $SLURM_ARRAY_TASK_MIN = 1


source /opt/uochb/soft/spack/20211108-git/share/spack/setup-env.sh 
spack load julia
spack load parallel

doit() {
    # set 1 # associative
    # time julia get-lattice.jl $1 $2 -1.0  -1.0   3.0   0.0   0.0   0.0
    
    # set 2 # segregative case
    # time julia get-lattice.jl $1 $2  1.0   1.0  -3.0   0.0   0.0   0.0
    
    # set 3 # counter-ionic
    # time julia get-lattice.jl $1 $2  2.0   0.0   0.5   0.0   0.0   0.0

    # set 4 # ATP
    time julia get-lattice.jl $1 $2 -2.94081 -1.00765 0.932429 -0.380839 -0.27919 -0.889506
    
}
export -f doit

param_store=param_product.txt  # args.txt contains 1000 lines with 2 arguments per line.

param_a=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')    # Get first argument
param_b=$(cat $param_store | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $2}')    # Get second argument

doit $param_a $param_b
