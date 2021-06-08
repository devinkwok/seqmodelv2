from argparse import ArgumentParser
import os
import abc
import typing
from seqmodel import hparam
from seqmodel.run import Initializer
from test import find_subclasses


FLOAT_SIG_DIGITS = 3
"""FLOAT_SIG_DIGITS is number of significant digits to round floats
when generating canonical string from hparams.
Floats (therefore hparams) are considered equal if they differ by less than
FLOAT_SIG_DIGITS many significant digits, causing job paths to be identical.
"""


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
    # define internal function for formatting string
    def fill_category(category: str, names_and_prefixes: typing.List[typing.Tuple[str, str]]) -> str:

        def fill_hparam(name: str, prefix: str):
            if name not in changed_hparams:
                return ''
            value = changed_hparams[name]
            if value is None:
                value = 'None'
            elif type(value) == bool:
                value = 'T' if value else 'F'
            elif type(value) == float:
                # use string format code to return scientific notation
                # with FLOAT_SIG_DIGITS - 1 decimals
                value = ("%." + str(FLOAT_SIG_DIGITS - 1) + "E") % (value)
            elif type(value) == str:
                # replace os.sep with pipe
                value = value.replace(os.sep, '|') + '_'
                if prefix != '': # add separator between prefix and str 
                    prefix = prefix + '='
            return prefix + str(value)

        out_str = []  # otherwise include category name
        for name, prefix in names_and_prefixes:
            out_str += [fill_hparam(name, prefix)]
        out_str = ''.join(out_str)
        if len(out_str) == 0:
            return ''
        return category + '=' + out_str

    #TODO use run.py or file based approach to get argparsers?
    # get non-default hparams
    # default_hparams = Initializer.get_parser(hparams)
    # hparams = hparam.parse_dict(hparams)  # check that hparams are valid by parsing them
    # changed_hparams = hparam.changed_hparams(hparams, default_hparams)

    default_hparams = ArgumentParser()
    for module in find_subclasses(hparam.Hparams, search_paths=[
        'seqmodel/dataset/', 'seqmodel/model/', 'seqmodel/task/', 'seqmodel/run.py'],
        exclude=[hparam.Hparams]):
        default_hparams = module._default_hparams(default_hparams)
    hparams = hparam.parse_dict(hparams, default_hparams)  # check that hparams are valid by parsing them
    changed_hparams = hparam.changed_hparams(hparams, default_hparams)
    # generate canonical string
    data_str = ''.join([
        fill_category('seq', [
            ('seq_path', ''), 
            ('train_intervals', 'int'),
            ('valid_intervals', 'int_v'),
            ('test_intervals', 'int_t'),
        ]),
        fill_category('mat', [
            ('train_mat', ''),
            ('valid_mat', 'v'),
            ('test_mat', 't'),
        ]),
    ])
    model_str = ''.join([
        fill_category('n', [
            ('n_layers', ''), 
            ('n_heads', 'x'), 
            ('n_decode_layers', 'dec'),
            ('sum_representation', 'sum'),
        ]),
        fill_category('d', [
            ('n_repr_dims', ''), 
            ('n_feedforward_dims', 'x'), 
        ]),
        fill_category('drop', [
            ('dropout', ''), 
            ('posencoder_dropout', 'pos'),
            ('decode_dropout', 'dec'),
        ]),
    ])
    task_str = ''.join([
        fill_category('pt', [
            ('mask_prop', 'm'),
            ('keep_prop', 'k'),
            ('random_prop', 'r'),
            ('val_mask_prop', 'mv'),
            ('val_keep_prop', 'kv'),
            ('val_random_prop', 'rv'),
        ]),
        fill_category('lr', [
            ('lr', ''),
        ]),
        fill_category('b', [
            ('batch_size', ''),
            ('accumulate_grad_batches', 'x'),
            ('valid_batch_size', 'v'),
            ('test_batch_size', 't'),
        ]),
        fill_category('opt', [
            ('adam_beta_1', 'b1'),
            ('adam_beta_2', 'b2'),
            ('adam_eps', 'e'),
            ('weight_decay', 'wd'),
            ('gradient_clip_val', 'clip'),
        ]),
    ])
    version_str = ''.join([
        fill_category('v', [
            ('init_version', ''),
            ('init_task', '-'),
            ('init_mode', '.'),
            ('load_encoder_from_checkpoint', 'loadenc'),
        ]),
    ])
    if data_str == '':
        data_str = 'default_data'
    if model_str == '':
        model_str = 'default_model'
    if task_str == '':
        task_str = 'default_task'
    if version_str == '':
        version_str = 'default_version'
    return [data_str, model_str, task_str, version_str]


class Job(hparam.Hparams, abc.ABC):
    """Represents an interface with a device for running training/inference.
    Uses `hparam.Hparams` interface to store parameters, however only the
    parameters defined by `run.py` are used to define model training/inference.
    """
    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--job_out_dir', default='out', type=str,
                            help='base path for storing job scripts and outputs')
        parser.add_argument('--gpus', default=1, type=int,
                            help='number of gpus, use cpu if 0')
        parser.add_argument('--num_nodes', default=1, type=int,
                            help='TODO')
        parser.add_argument('--num_processes', default=1, type=int,
                            help='TODO')
        parser.add_argument('--resume_from_checkpoint', default=None, type=str,
                            help='TODO')
        parser.add_argument('--max_epochs', default=1000, type=int,
                            help='TODO')
        parser.add_argument('--max_steps', default=None, type=int,
                            help='TODO')
        parser.add_argument('--terminate_on_nan', default=True, type=hparam.str2bool,
                            help='TODO')
        parser.add_argument('--track_grad_norm', default=-1, type=int,
                            help='TODO')
        parser.add_argument('--limit_train_batches', default=1.0, type=float,
                            help='TODO')
        parser.add_argument('--limit_val_batches', default=1.0, type=float,
                            help='TODO')
        parser.add_argument('--limit_test_batches', default=1.0, type=float,
                            help='TODO')
        parser.add_argument('--deterministic', default=False, type=hparam.str2bool,
                            help='TODO')
        parser.add_argument('--check_val_every_n_epoch', default=1, type=int,
                            help='TODO')
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
        Defines `--default_root_dir` arg for pytorch_lightning.

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
        Uses '--fast_dev_run=True' if testing.

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
        parser.add_argument('--local_env_command', default='source env/bin/activate', type=str,
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
        """Runs job script locally in shell

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
