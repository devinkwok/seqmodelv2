#!/bin/bash
#SBATCH --job-name={JOB_slurm_jobname}
#SBATCH --account={JOB_slurm_account}
#SBATCH --time={JOB_slurm_time}
#SBATCH --cpus-per-task={JOB_slurm_cpus_per_task}
#SBATCH --gres=gpu:{JOB_slurm_gpu_type}:{JOB_slurm_n_gpu}
#SBATCH --mem={JOB_slurm_mem}
{JOB_slurm_stdout}
{JOB_slurm_array}

## load modules
module load nixpkgs/16.09  gcc/7.3.0 cuda/10.1 cudnn/7.6.5 python/3.7.4

## setup virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

## install project dependencies
pip install --no-index --upgrade pip
pip install --no-index -r ./requirements.txt

## checkout code to localscratch
cd {JOB_src_dir}
git worktree add $SLURM_TMPDIR {JOB_version}

## copy and extract data to localscratch
mkdir $SLURM_TMPDIR/data
cp {JOB_data_dir} $SLURM_TMPDIR/data/

## extract all .tar.gz and .gz data files
tar xzvf $SLURM_TMPDIR/data/*.tar.gz
gunzip $SLURM_TMPDIR/data/*.gz

## set working dir to src root
cd $SLURM_TMPDIR

## run job
{JOB_commands}

## clean up by stopping virtualenv
deactivate
