from argparse import ArgumentParser
import os
import abc
import typing
from seqmodel.hparam import Hparams


class Job(Hparams, abc.ABC):
    """Represents an interface with a device for running training/inference.
    Uses `hparam.Hparams` interface to store parameters, however only the
    parameters defined by `run.py` are used to define model training/inference.
    """
    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--job_output_root_dir', default='./out', type=str,
                            help='base path for storing job scripts and outputs')
        return parser

    @staticmethod
    def hparams_to_canonical_str(hparams: dict)-> typing.List[str]:
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

    def path_to_latest_checkpoint(self, base_path: str)-> os.PathLike:
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

    def fill_latest_ckpt_paths(self, hparams: dict)-> dict:
        """Replaces any `[base_path]/$LATEST_CHECKPOINT`
        with `path_to_latest_checkpoint([base_path])`

        Args:
            base_path (str): root directory to search from.

        Returns:
            os.PathLike: path to most recent checkpoint file
        """
        return None #TODO

    def make_script(self, filepath: str, hparams: dict):
        """Generates job script and saves to filepath.
        Calls `fill_latest_ckpt_paths` on hparams.
        Creates new replicate if same job has been run previously at same log_path.

        Args:
            filepath (str): location of script, replicate number is appended if needed
            hparams (dict): hparams for run.py

        Raises:
            ValueError: if hyperparameters are not valid
        """
        raise ValueError() #TODO

    @abc.abstractmethod
    def _make_script(self, hparams: dict)-> str:
        """Returns script to write to file (defined by subclass).

        Args:
            hparams (dict): hparams for run.py

        Returns:
            str: script
        """
        return None #TODO

    @abc.abstractmethod
    def submit(self, hparams: dict)-> str:
        """Generate and run job script (defined by subclass).

        Args:
            hparams (dict): hyperparameters

        Returns:
            str: path to script and outputs
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

    def _make_script(self, hparams: dict)-> str:
        """Uses `str.format()` to fill in `template_local_shell.sh`.

        Args:
            hparams (dict): hparams for run.py

        Returns:
            str: filled in `template_local_shell.sh`
        """
        return None #TODO

    def submit(self, hparams: dict)-> str:
        """Generates job script and runs it locally in shell

        Args:
            hparams (dict): hyperparameters (raise exception)

        Returns:
            str: local path to script and outputs
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

    def _make_script(self, hparams: dict)-> str:
        """Uses `str.format()` to fill in `template_remote_slurm.sh`.

        Args:
            hparams (dict): hparams for run.py

        Returns:
            str: filled in `template_local_shell.sh`
        """
        return None #TODO

    def submit(self, hparams: dict)-> str:
        """Submits job via Slurm over ssh.

        Args:
            hparams (dict): hyperparameters (raise exception)

        Returns:
            str: remote path to script and outputs
        """
        return None #TODO
