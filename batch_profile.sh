#!/bin/bash
#
#SBATCH --job-name=ridge-profile
#SBATCH --output=output.log
#
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

# The above configs do the following
# 	job-name: some human-readable string for a name
#	output: Place where all of STDOUT+STDERR goes
#	ntasks: Tells SLURM the number of instances of this procedure are needed
#		By default, each instance of the procedure is allocated a single processor
#		If each task is itself multithreaded, then this will not do, and you
#		should also specify 'cpus-per-task'
#	cpus-per-task: The number of processors needed to compute one task.
#		So if each task does something like set OMP_NUM_THREADS=8,
#		then this option should probably be set to 8
#	time: The time until a task is killed due to timeout 

# This SLURM batch file is for profiling the code. If you simply want to run the
# code, use batch.sh instead


# Weird round-about method of loading modules on 
# stampede that allows for proper mpi4py usage
# Will probably be different in GallantLab
#module load python/2.7.3-epd-7.3.2   # this is the older one...
#module load mpi4py                   # a second step is needed
module load intel/14.0.1.106
module load python/2.7.6  
module load cuda/6.0
export PYTHONPATH="/opt/apps/python/epd/7.3.2/modules/lib/python:/opt/apps/python/epd/7.3.2/lib:$HOME/.python/lib/python2.7/site-packages:$HOME/lib64/python2.7/site-packages"
export PATH="$PATH:/opt/apps/intel14/mvapich2_2_0/python/2.7.6/lib/python2.7/site-packages/mpi4py/bin/"
export OMP_NUM_THREADS=8
# use ibrun on stampede in order to use MPI
# this will need to change when running in GallantLab
ibrun python -m cProfile -o test.pstats ./test.py --benchmark 
