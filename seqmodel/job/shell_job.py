import os
from seqmodel import Hparams
from seqmodel.job.abstract_job import Job


class ShellJob(Job):
    """Run job from shell with preinstalled python environment.
    """
    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--activate_env_command', default='source env/bin/activate', type=str,
                            help='command to start preinstalled python environment')
        parser.add_argument('--deactivate_env_command', default='deactivate', type=str,
                            help='command to stop preinstalled python environment')
        return parser

    def _create(self, hparams: dict) -> str:
        """Uses `str.format()` to fill in `template_shell.sh`.

        Args:
            hparams (dict): hparams for run.py

        Returns:
            str: filled in `template_shell.sh`
        """
        args = Hparams.to_args(hparams)
        root_dir = hparams['default_root_dir']
        stdout_file = self.os.join(root_dir, self.STDOUT_FILENAME)
        stderr_file = self.os.join(root_dir, self.STDERR_FILENAME)
        command_str = f'python seqmodel/run.py {args}'

        script = self.template.format(
                JOB_local_env_activate=self.hparams.activate_env_command,
                JOB_commands=command_str,
                JOB_local_env_deactivate=self.hparams.deactivate_env_command,
            )
        return script

    def submit(self, path_to_job_script: os.PathLike) -> str:
        """Runs job script locally in shell

        Args:
            path_to_job_script (os.PathLike): location of job script relative to `job_out_dir`

        Returns:
            str: status of job
        """
        replicate_path, _ = self.os.split(path_to_job_script)
        stdout_file = self.os.join(replicate_path, self.STDOUT_FILENAME)
        stderr_file = self.os.join(replicate_path, self.STDERR_FILENAME)
        self.os.command(f'chmod +r {path_to_job_script}')
        self.os.command(f'sh {path_to_job_script} > {stdout_file} 2> {stderr_file} &')
        return None #TODO

    template = \
"""
## activate python environment
{JOB_local_env_activate}

## run model
{JOB_commands}

## deactivate python environment
{JOB_local_env_deactivate}
"""
