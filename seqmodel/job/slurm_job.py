import os
from datetime import timedelta
from seqmodel.job.abstract_job import Job
from seqmodel.job.abstract_job import JobHparams


class SlurmJobHparams(JobHparams):
    def _default_hparams():
        return {
            'remote_ssh_login': (None, 'ssh', str,
                'ssh credentials in the form [username@remote]'),
            'pull_src_dir': ('~/proj/src', 'srcdir', str,
                'location of source repository on remote'),
            'pull_data_dir': ('~/data', 'datadir', str,
                'location of predownloaded data on remote'),
            'slurm_account': (None, 'sacc', str,
                'name of billing account'),
            'slurm_time': (1., 'stime', float,
                'max runtime in hours (can be fractional)'),
            'slurm_cpus': (1, 'scpu', int,
                'number of cpus per node'),
            'slurm_mem': (8000, 'smem', int,
                'memory per node (mb)'),
            'slurm_gpu_type': ('p100', 'sgpu', str,
                'type of gpu to request (number is filled from hparam)'),
        }

class SlurmJob(Job):

    def __init__(self, os_interface, **hparams):
        """Modifies `Job` interface to use Slurm job manager.
        Note: change self.template to support multiple script templates.
        """
        super().__init__(os_interface, SlurmJobHparams(**hparams))
        self.template = CEDAR_TEMPLATE

    @staticmethod
    def format_slurm_time(hours: float) -> str:
        """Returns time in format "days-hours:minutes:seconds".

        Args:
            hours (float): time in hours

        Returns:
            str: time in days-hours:minutes:seconds
        """
        t = timedelta(hours=hours)
        h, m_s = divmod(t.seconds, 3600)
        m, s = divmod(m_s, 60)
        return f'{str(t.days).zfill(2)}-{str(h).zfill(2)}:{str(m).zfill(2)}:{str(s).zfill(2)}'

    def _create(self, hparams: dict) -> str:
        """Uses `str.format()` to fill in `templateslurm.sh`.

        Args:
            hparams (dict): hparams for run.py

        Returns:
            str: filled in `template_slurm.sh`
        """
        args = Hparams.to_args(hparams)
        command_str = f'python seqmodel/run.py {args}'

        script = self.template.format(
                JOB_slurm_jobname=hparams['default_root_dir'],
                JOB_slurm_account=self.hparams.slurm_account,
                JOB_slurm_time=self.format_slurm_time(self.hparams.slurm_time),
                JOB_slurm_cpus_per_task=self.hparams.slurm_cpus,
                JOB_slurm_n_gpu=hparams['gpus'],
                JOB_slurm_mem=self.hparams.slurm_mem,
                JOB_slurm_stdout=self.STDOUT_FILENAME,
                JOB_slurm_stderr=self.STDERR_FILENAME,
                JOB_src_dir=self.hparams.pull_src_dir,
                JOB_version=hparams['init_version'],
                JOB_data_dir=self.hparams.pull_data_dir,
                JOB_commands=command_str,
            )
        return script

    def submit(self, path_to_job_script: os.PathLike) -> str:
        """Submits job via Slurm.

        Args:
            path_to_job_script (os.PathLike): location of job script relative to `job_out_dir`

        Returns:
            str: Slurm job id number and status
        """
        self.os.command(f'cd {path_to_job_script}')
        self.os.command(f'sbatch {self.SCRIPT_NAME}')
        return None #TODO


CEDAR_TEMPLATE = \
"""
#!/bin/bash
#SBATCH --job-name={JOB_slurm_jobname}
#SBATCH --account={JOB_slurm_account}
#SBATCH --time={JOB_slurm_time}
#SBATCH --cpus-per-task={JOB_slurm_cpus_per_task}
#SBATCH --gres=gpu:{JOB_slurm_gpu_type}:{JOB_slurm_n_gpu}
#SBATCH --mem={JOB_slurm_mem}
#SBATCH --output={JOB_slurm_stdout}
#SBATCH --error={JOB_slurm_stderr}

## load modules
module load nixpkgs/16.09 gcc/7.3.0 cuda/10.2 cudnn/7.6.5 python/3.7.4

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
"""
