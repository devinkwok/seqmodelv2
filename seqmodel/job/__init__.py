from argparse import ArgumentParser
import os
import re
import abc
import typing
from datetime import timedelta
from seqmodel import hparam
from seqmodel.run import Initializer
from seqmodel.job.os_interface import *


# Number of significant digits to round floats for canonical hparam string.
# Floats (therefore hparams) are considered equal if they differ by less
# precision than FLOAT_SIG_DIGITS, making their canonical paths identical.
FLOAT_SIG_DIGITS = 3

# Pad replicate directories (which are integers) with zeros up to this length.
# This allows UNIX-style directory sorting, as otherwise `10` would come before `2`.
N_REPLICATE_ZEROS = 2

# Keyword for replacement by `_replace_latest_ckpt_paths`
LATEST_CKPT_SHORTHAND = 'JOB_latest_checkpoint'

# Default script and shell output filenames in canonical/replicate path.
SCRIPT_NAME = 'job.sh'
STDOUT_FILENAME = 'job.out'
STDERR_FILENAME = 'job.err'


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

    def __init__(self, os_interface: OsInterface, **hparams):
        """Initializes Job.
        Args:
            os_interface (OsInterface): Object defining how `Job` should
                read filesystem and call commands (e.g. locally or remotely).
        """
        self.os = os_interface
        super().__init__(**hparams)

    def _fill_category(
        self, hparams: dict,
        category: str,
        names_and_prefixes: typing.List[typing.Tuple[str, str]]
    ) -> str:
        """Internal function defining formatting for `hparams_to_canonical_str`.
        Changes display of some types as follows:
        - None becomes 'None'
        - bool becomes 'T' or 'F'
        - float is displayed in scientific notation rounded to FLOAT_SIG_DIGITS - 1
        - str has `self.os.sep` replaced with pipe '|', and is wrapped with '=' and '_'.

        Args:
            category (str): prefix name
            names_and_prefixes (typing.List[typing.Tuple[str, str]]):
                List of tuples of form (hparam_key, canonical_shorthand).

        Returns:
            str: '' if no hparam_key is in hparams, else str in the format
                '{category}={shorthand0}{value0}{shorthand1}{value1}...'
        """
        def fill_hparam(hparams: dict, name: str, prefix: str):
            if name not in hparams:
                return ''
            value = hparams[name]
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
                value = value.replace(self.os.sep, '|') + '_'
                if prefix != '': # add separator between prefix and str 
                    prefix = prefix + '='
            return prefix + str(value)

        out_str = []
        for name, prefix in names_and_prefixes:
            out_str += [fill_hparam(hparams, name, prefix)]
        out_str = ''.join(out_str)
        if len(out_str) == 0:
            return ''
        return category + '=' + out_str

    def hparams_to_canonical_str(self, hparams: dict) -> typing.List[str]:
        """Converts dict of hyperparameters into ordered list of strings.
        Only non-default hyperparameters are recorded.
        Hparams are arranged by group as model, dataset, task, then in lexical order.
        Each hparam value is converted to a canonical string.
        The resulting list of strings can be concatenated to form a descriptive name,
        or turned into the canonical path:
        `[version]/[model hparams]/[dataset hparams]/[task hparams]/`.

        Args:
            hparams (dict): valid non-default hparams

        Returns:
            list[str]: canonical string form of hparams in canonical order,
                apply self.os.join() to get canonical path
        """
        default_hparams = ArgumentParser()
        for module in hparam.find_subclasses(hparam.Hparams, search_paths=[
            'seqmodel/dataset/', 'seqmodel/model/', 'seqmodel/task/', 'seqmodel/run.py'],
            exclude=[hparam.Hparams]):
            default_hparams = module._default_hparams(default_hparams)
        changed_hparams = hparam.changed_hparams(hparams, default_hparams)
        # generate canonical string
        version_str = ''.join([
            self._fill_category(changed_hparams, 'v', [
                ('init_version', ''),
                ('init_task', 't'),
                ('init_mode', 'm'),
                ('load_encoder_from_checkpoint', 'loadenc'),
            ]),
        ])
        data_str = ''.join([
            self._fill_category(changed_hparams, 'seq', [
                ('seq_path', ''), 
                ('train_intervals', 'int'),
                ('valid_intervals', 'int_v'),
                ('test_intervals', 'int_t'),
            ]),
            self._fill_category(changed_hparams, 'mat', [
                ('train_mat', ''),
                ('valid_mat', 'v'),
                ('test_mat', 't'),
            ]),
        ])
        model_str = ''.join([
            self._fill_category(changed_hparams, 'n', [
                ('n_layers', ''), 
                ('n_heads', 'x'), 
                ('n_decode_layers', 'dec'),
                ('sum_representation', 'sum'),
            ]),
            self._fill_category(changed_hparams, 'd', [
                ('n_repr_dims', ''), 
                ('n_feedforward_dims', 'x'), 
            ]),
            self._fill_category(changed_hparams, 'drop', [
                ('dropout', ''), 
                ('posencoder_dropout', 'pos'),
                ('decode_dropout', 'dec'),
            ]),
        ])
        task_str = ''.join([
            self._fill_category(changed_hparams, 'pt', [
                ('mask_prop', 'm'),
                ('keep_prop', 'k'),
                ('random_prop', 'r'),
                ('val_mask_prop', 'mv'),
                ('val_keep_prop', 'kv'),
                ('val_random_prop', 'rv'),
            ]),
            self._fill_category(changed_hparams, 'lr', [
                ('lr', ''),
            ]),
            self._fill_category(changed_hparams, 'b', [
                ('batch_size', ''),
                ('accumulate_grad_batches', 'x'),
                ('valid_batch_size', 'v'),
                ('test_batch_size', 't'),
            ]),
            self._fill_category(changed_hparams, 'opt', [
                ('adam_beta_1', 'b1'),
                ('adam_beta_2', 'b2'),
                ('adam_eps', 'e'),
                ('weight_decay', 'wd'),
                ('gradient_clip_val', 'clip'),
            ]),
        ])
        if version_str == '':
            version_str = 'default_version'
        if data_str == '':
            data_str = 'default_data'
        if model_str == '':
            model_str = 'default_model'
        if task_str == '':
            task_str = 'default_task'
        return [version_str, data_str, model_str, task_str]

    def list_checkpoints_by_iter(self, base_path: str) -> typing.List[os.PathLike]:
        """Finds filepath of most recent checkpoint by highest epoch/iteration
        numbers in filename. Searches all subdirectories of given base_path recursively,
        relative to current working directory.
        Does not open checkpoint file to verify any iteration values.
        Simply sorts by the last 2 integers present in the filename,
        assuming they are epoch and iteration respectively.

        Args:
            base_path (str): root directory to search from.

        Returns:
            os.PathLike: path to most recent checkpoint file
        """
        checkpoints = []
        for path, ckpt_name in self.os.find(base_path, suffix='.ckpt'):
            integers = [int(i) for i in re.findall(r'\d+', ckpt_name)]
            checkpoint_path = self.os.join(path, ckpt_name)
            if len(integers) < 2:
                warnings.warn(f'Listing checkpoints at {base_path}, ' +
                    f'checkpoint {checkpoint_path} epoch/iter missing?')
            else:
                epochs = integers[-2]
                iters = integers[-1]
                checkpoints.append((checkpoint_path, epochs, iters))
        return sorted(checkpoints, key=lambda tup: (tup[1], tup[2], tup[0]))

    def _replace_latest_ckpt_paths(self, hparams: dict) -> dict:
        """Replaces any `[base_path]/JOB_latest_checkpoint`
        with latest checkpoint in `sort_checkpoints([base_path])`

        Args:
            base_path (str): root directory to search from.

        Returns:
            os.PathLike: path to most recent checkpoint file
        """
        new_hparams = dict(hparams)
        for k, v in hparams.items():
            if type(v) is str and v.endswith(LATEST_CKPT_SHORTHAND):
                base_path = v[:-len(LATEST_CKPT_SHORTHAND)]
                checkpoints_by_iter = self.list_checkpoints_by_iter(base_path)
                latest_ckpt_path, _, _ = checkpoints_by_iter[-1]
                print(f'Replaced hparam `{k}` with {latest_ckpt_path}, ' +
                    f'original value was {v}')
                new_hparams[k] = latest_ckpt_path
        return new_hparams

    def replicates(self, base_path: os.PathLike, status='all') -> typing.List[os.PathLike]:
        """Looks in `path` for replicates.
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
            os.PathLike: path to new replicate subdirectory including `job_out_dir`,
                replicate number guaranteed to be largest under canonical_path
        """
        # create canonical_path if not existing
        full_path = self.os.join(self.hparams.job_out_dir, canonical_path)
        self.os.mkdir(full_path)
        # look for existing integer dirs
        last_replicate = 0
        for f in self.os.list(full_path, suffix=self.os.sep):
            try:
                replicate_number = int(f[:-1])
            except ValueError:
                continue
            if last_replicate < replicate_number:
                last_replicate = replicate_number
        # increment integer dirs, left pad with zeros
        next_replicate = str(last_replicate + 1).zfill(N_REPLICATE_ZEROS)
        new_replicate_path = self.os.join(
            self.hparams.job_out_dir,
            canonical_path,
            next_replicate)
        self.os.mkdir(new_replicate_path)
        return new_replicate_path

    def create(self, hparams: dict):
        """Generates job script.
        Checks hparams using `get_parser` in run.py
        Calls `hparams_to_canonical_str` to define output location.
        Defines `--default_root_dir` using canonical path.
        Defines other pytorch_lightning args from self.hparams.
        Calls `new_replicate` to create job path.
        Calls `self._create` to create job script.

        Args:
            hparams (dict): hparams for run.py

        Returns:
            os.PathLike: path to job script relative to `job_out_dir`
        """
        # check hparams are valid in run.py and find canonical_path
        parser = Initializer.get_parser(hparams)
        hparams = hparam.parse_dict(hparams, parser)
        hparams = hparam.changed_hparams(hparams, parser)
        canonical_str = self.hparams_to_canonical_str(hparams)
        canonical_path = self.os.join(*canonical_str)
        replicate_path = self.new_replicate(canonical_path)
        # update hparams with job args, canonical_path, and latest ckpts
        job_params = hparam.changed_hparams(self.hparams, self.default_hparams())
        hparams = {**hparams, **job_params,
            'default_root_dir': replicate_path}
        hparams = self._replace_latest_ckpt_paths(hparams)
        # create and save script
        script = self._create(hparams)
        script_path = self.os.join(replicate_path, SCRIPT_NAME)
        self.os.write(script, script_path)
        return script_path

    @abc.abstractmethod
    def _create(self, hparams: dict) -> str:
        """Returns script to write to file (defined by subclass).

        Args:
            hparams (dict): hparams for run.py

        Returns:
            str: script
        """
        return None

    @abc.abstractmethod
    def submit(self, path_to_job_script: os.PathLike) -> str:
        """Run job script (defined by subclass).

        Args:
            path_to_job_script (os.PathLike): location of job script relative to `job_out_dir`

        Returns:
            str: status of job
        """
        return None


class ShellJob(Job):
    """Run job from shell with preinstalled python environment.
    """
    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--activate_env_command', default='source env/bin/activate', type=str,
                            help='command to start preinstalled python environment')
        return parser

    def _create(self, hparams: dict) -> str:
        """Uses `str.format()` to fill in `template_shell.sh`.

        Args:
            hparams (dict): hparams for run.py

        Returns:
            str: filled in `template_shell.sh`
        """
        args = hparam.to_args(hparams)
        root_dir = hparams['default_root_dir']
        stdout_file = self.os.join(root_dir, STDOUT_FILENAME)
        stderr_file = self.os.join(root_dir, STDERR_FILENAME)
        command_str = f'python seqmodel/run.py {args}'

        with open('seqmodel/job/template_shell.sh', 'r') as f:
            template = f.read()
        script = template.format(
                JOB_local_env_activate=self.hparams.activate_env_command,
                JOB_commands=command_str,
            )
        return script

    def submit(self, path_to_job_script: os.PathLike) -> str:
        """Runs job script locally in shell

        Args:
            path_to_job_script (os.PathLike): location of job script relative to `job_out_dir`

        Returns:
            str: status of job
        """
        replicate_path, script_file = self.os.split(path_to_job_script)
        stdout_file = self.os.join(replicate_path, STDOUT_FILENAME)
        stderr_file = self.os.join(replicate_path, STDERR_FILENAME)
        self.os.command(f'chmod +r {path_to_job_script}')
        print(f'sh {path_to_job_script} > {stdout_file} 2> {stderr_file}')
        self.os.command(f'sh {path_to_job_script} > {stdout_file} 2> {stderr_file}')
        return None #TODO


class SlurmJob(Job):
    """Modifies `Job` interface to use Slurm job manager.
    """
    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--remote_ssh_login', default=None, type=str,
                            help='ssh credentials in the form [username@remote]')
        parser.add_argument('--pull_src_dir', default='~/proj/src', type=str,
                            help='location of source repository on remote')
        parser.add_argument('--pull_data_dir', default='~/data', type=str,
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
        """Uses `str.format()` to fill in `template_slurm.sh`.

        Args:
            hparams (dict): hparams for run.py

        Returns:
            str: filled in `template_slurm.sh`
        """
        args = hparam.to_args(hparams)
        command_str = f'python seqmodel/run.py {args}'

        with open('seqmodel/job/template_slurm.sh', 'r') as f:
            template = f.read()
        script = template.format(
                JOB_slurm_jobname=hparams['default_root_dir'],
                JOB_slurm_account=self.hparams.slurm_account,
                JOB_slurm_time=self.format_slurm_time(self.hparams.slurm_time),
                JOB_slurm_cpus_per_task=self.hparams.slurm_cpus,
                JOB_slurm_n_gpu=hparams['gpus'],
                JOB_slurm_mem=self.hparams.slurm_mem,
                JOB_slurm_stdout=STDOUT_FILENAME,
                JOB_slurm_stderr=STDERR_FILENAME,
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
        self.os.command(f'sbatch {SCRIPT_NAME}')
        return None #TODO
