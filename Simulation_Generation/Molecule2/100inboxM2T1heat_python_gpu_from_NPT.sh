#!/bin/bash -l
#SBATCH --output=/scratch_tmp/prj/ch_smiecs_epsrc/outputs%j.out
#SBATCH --mem=48G
#SBATCH --gres=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=darvin.magundayao@kcl.ac.uk
#SBATCH --time=2-0:00
#SBATCH --nodes=1
module load cuda/10.0.130-gcc-13.2.0
module load gromacs/2021.5-gcc-11.4.0-cuda-11.8.0
python3 100inboxM2T1heat_python_gpu_from_NPT.py

