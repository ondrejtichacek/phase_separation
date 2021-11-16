#!/bin/bash
#SBATCH -J lattice
#SBATCH --qos=matlab
#SBATCH --partition=cpu,scpu,bfill
#SBATCH --time=0-12:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=90G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ondrej.tichacek@uochb.cas.cz

source /opt/uochb/soft/spack/20211108-git/share/spack/setup-env.sh 
spack load julia
spack load parallel

doit() {
    time julia get-lattice.jl $1 $2
}
export -f doit

# Parallel notes
# `--header : ` takes the first value in a column as name
# `--results ./$SLURM_JOB_ID ` outputs stdout/stderr to a specified folder
# X="0.01       0.06210526 0.11421053 0.16631579 0.21842105 0.27052632 0.32263158 0.37473684 0.42684211 0.47894737 0.53105263 0.58315789 0.63526316 0.68736842 0.73947368 0.79157895 0.84368421 0.89578947 0.94789474 1."

# python gen-params.py > params.txt

parallel -j 36 \
    doit \
    :::: params.txt \
    :::: params.txt

#    --header : \
#      ::: xb $X \
#      ::: xr $X
# --results ./$SLURM_JOB_ID \
