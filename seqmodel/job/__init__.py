from argparse import ArgumentParser
import os
import abc
import typing
from seqmodel.hparam import Hparams


def hparams_to_canonical_str(hparams: dict) -> typing.List[str]:
    """Converts dict of hyperparameters into ordered list of strings.
    Only non-default hyperparameters are recorded,
    default values are defined by `seqmodel.run.get_parser()`.
    Hparams are arranged by group as model, dataset, task, then in lexical order.
    Each hparam value is converted to a canonical string.
    The resulting list of strings can be concatenated to form a descriptive name,
    or turned into the canonical path:
    `[version]/[model hparams]/[dataset hparams]/[task hparams]/`.

    Args:
        hparams (dict): valid non-default hparams

    Returns:
        list[str]: canonical string form of hparams in canonical order,
            apply os.path.join() to get canonical path
    """
    return None #TODO


class Job(Hparams, abc.ABC):
    """Represents an interface with a device for running training/inference.
    Uses `hparam.Hparams` interface to store parameters, however only the
    parameters defined by `run.py` are used to define model training/inference.
    """
    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--job_out_dir', default='./out', type=str,
                            help='base path for storing job scripts and outputs')
        return parser

    def path_to_latest_checkpoint(self, base_path: str) -> os.PathLike:
        """Finds filepath of most recent checkpoint by highest epoch/iteration
        number given base_path. Searches all subdirectories recursively.
        Searches from current working directory in local or remote
        depending on remote_type.

        Args:
            base_path (str): root directory to search from.

        Returns:
            os.PathLike: path to most recent checkpoint file
        """
        return None #TODO

    def replicates(self, canonical_path: os.PathLike, status='all') -> typing.List[os.PathLike]:
        """Looks in `job_out_dir/canonical_path` for replicates.
        Returns list of job script filenames for replicates in the category `status`.
        `canonical_path` may not exist, in which case an empty list is returned.

        Args:
            canonical_path (os.PathLike): base path to search for replicates.
            status (str): category of replicate to find, one of
                {'all', 'created', 'running', 'error', 'timeout', 'complete'}

        Returns:
            dict[str, os.PathLike]: dict of job status categories and lists of paths to jobs
        """
        return None #TODO

    def new_replicate(self, canonical_path: os.PathLike) -> os.PathLike:
        """Checks if `canonical_path` exists in `job_out_dir`.
        If non-existing, creates subdirectories as needed.
        Else, finds highest replicate number and
        creates/returns new subdirectory which increments the replicate number.

        Args:
            canonical_path (os.PathLike): base path to search for replicates.

        Returns:
            os.PathLike: path to new replicate subdirectory,
                replicate number guaranteed to be largest under canonical_path
        """

    def _fill_latest_ckpt_paths(self, hparams: dict) -> dict:
        """Replaces any `[base_path]/$LATEST_CHECKPOINT`
        with `path_to_latest_checkpoint([base_path])`

        Args:
            base_path (str): root directory to search from.

        Returns:
            os.PathLike: path to most recent checkpoint file
        """
        return None #TODO

    def create(self, hparams: dict):
        """Generates job script.
        Calls `hparams_to_canonical_str` to get job path.
        Calls `new_replicate` to increment replicates or create job path.
        Calls `fill_latest_ckpt_paths` on hparams.

        Args:
            hparams (dict): hparams for run.py

        Returns:
            os.PathLike: path to job script relative to `job_out_dir`
        """
        return None #TODO

    @abc.abstractmethod
    def _create(self, hparams: dict) -> str:
        """Returns script to write to file (defined by subclass).

        Args:
            hparams (dict): hparams for run.py

        Returns:
            str: script
        """
        return None #TODO

    @abc.abstractmethod
    def submit(self, path_to_job_script: os.PathLike) -> str:
        """Run job script (defined by subclass).

        Args:
            path_to_job_script (os.PathLike): location of job script relative to `job_out_dir`

        Returns:
            str: status of job
        """
        return None #TODO


class LocalJob(Job):
    """Run job from local shell with preinstalled python environment.
    """
    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--local_env_command', default='source ./env/bin/activate', type=str,
                            help='command to start preinstalled python environment')
        return parser

    def _create(self, hparams: dict) -> str:
        """Uses `str.format()` to fill in `template_local_shell.sh`.

        Args:
            hparams (dict): hparams for run.py

        Returns:
            str: filled in `template_local_shell.sh`
        """
        return None #TODO

    def submit(self, hparams: dict) -> str:
        """Generates job script and runs it locally in shell

        Args:
            path_to_job_script (os.PathLike): location of job script relative to `job_out_dir`

        Returns:
            str: status of job
        """
        return None #TODO


class RemoteSlurmJob(Job):
    """Modifies `Job` interface to use Slurm job manager over ssh.
    """
    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--remote_ssh_login', default=None, type=str,
                            help='ssh credentials in the form [username@remote]')
        parser.add_argument('--remote_src_dir', default='~/proj/src', type=str,
                            help='location of source repository on remote')
        parser.add_argument('--remote_data_dir', default='~/data', type=str,
                            help='location of predownloaded data on remote')
        parser.add_argument('--slurm_account', default=None, type=str,
                            help='name of billing account')
        parser.add_argument('--slurm_time', default=1., type=float,
                            help='max runtime in hours (can be fractional)')
        parser.add_argument('--slurm_cpus', default=1, type=int,
                            help='number of cpus per node')
        parser.add_argument('--slurm_mem', default=8000, type=int,
                            help='memory per node (mb)')
        parser.add_argument('--slurm_gpu_type', default='p100', type=str,
                            help='type of gpu to request (number is filled from hparam)')
        return parser

    def _create(self, hparams: dict) -> str:
        """Uses `str.format()` to fill in `template_remote_slurm.sh`.

        Args:
            hparams (dict): hparams for run.py

        Returns:
            str: filled in `template_local_shell.sh`
        """
        return None #TODO

    def submit(self, hparams: dict) -> str:
        """Submits job via Slurm over ssh.

        Args:
            path_to_job_script (os.PathLike): location of job script relative to `job_out_dir`

        Returns:
            str: Slurm job id number and status
        """
        return None #TODO
