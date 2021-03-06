import sys
import os
import re
import abc
import typing
import logging
from datetime import datetime
import torch
from seqmodel import VERSION
from seqmodel.hparam import Hparams
from seqmodel.hparam import HparamCollection
from seqmodel.run import Initializer
from seqmodel.run import Runner
from seqmodel.job.os_interface import OsInterface


class JobHparams(Hparams):
    def _default_hparams():
        return {
        'job_author': (None, 'n', str,
            'author of job for tracking submissions ' + \
            '(e.g. name of notebook, device, or person)'),
        'job_out_dir': ('out', 'dir', str,
            'base path for storing job scripts and outputs'),
        'gpus': (1, 'g', int,
            'number of gpus, use cpu if 0'),
        'num_nodes': (1, 'node', int,
            'TODO'),
        'num_processes': (1, 'proc', int,
            'TODO'),
        'resume_from_checkpoint': (None, 'cpload', str,
            'path to checkpoint to load (pytorch lightning)'),
        'max_epochs': (1000, 'mep', int,
            'maximum number of epochs to train (pytorch lightning)'),
        'max_steps': (None, 'mit', int,
            'maximum number of steps to train (pytorch lightning)'),
        'terminate_on_nan': (True, 'nan', bool,
            'stop training if NaN in weight/gradient (pytorch lightning)'),
        'track_grad_norm': (-1, 'gnorm', int,
            'type of norm (L1, L2, etc.) to save on gradients (pytorch lightning)'),
        'limit_train_batches': (1.0, 'bstr', float,
            'proportion (float) or number (int) of batches to train on'),
        'limit_val_batches': (1.0, 'bsval', float,
            'proportion (float) or number (int) of batches to validate on'),
        'limit_test_batches': (1.0, 'bste', float,
            'proportion (float) or number (int) of batches to test on'),
        'deterministic': (False, 'det', bool,
            'if True use deterministic training (fix random seed)'),
        'save_checkpoint_interval': (10000, 'cpsaveit', int,
            'number of steps to save checkpoint after'),
        'val_check_interval': (10000, 'valit', int,
            'number of steps to run validation after'),
        'reload_dataloaders_every_epoch': (True, 'dl', bool,
            'randomizes batch order (pytorch lightning)'),
        }

class Job(abc.ABC):
    """Represents an interface with a device for running training/inference.
    Uses `Hparams` interface to store parameters, however only the
    parameters defined by `run.py` are used to define model training/inference.
    """

    # Pad replicate directories (which are integers) with zeros up to this length.
    # This allows UNIX-style directory sorting, as otherwise `10` would come before `2`.
    N_REPLICATE_ZEROS = 2

    # Keyword for replacement by `_replace_latest_ckpt_paths`
    LATEST_CKPT_SHORTHAND = 'JOB_latest_checkpoint'

    # Default script and shell output filenames in canonical/replicate path.
    SCRIPT_NAME = 'job.sh'
    STDOUT_FILENAME = 'job.out'
    STDERR_FILENAME = 'job.err'

    SCRIPT_LAST_LINE = '# Job script auto-generated by {JOB_author} using {JOB_class} v{JOB_version} on {JOB_date}.'
    TIMEOUT_LAST_LINE = 'slurmstepd: error: \*\*\* JOB (\d+) ON (\w+) CANCELLED AT (.+) DUE TO TIME LIMIT \*\*\*'

    def __init__(self, os_interface: OsInterface, hparams: JobHparams):
        """Initializes Job.
        Args:
            hparams (JobHparams): hyperparameters (not tracked).
            os_interface (OsInterface): Object defining how `Job` should
                read filesystem and call commands (e.g. locally or remotely).
        """
        self.hparams = hparams
        self.os = os_interface
        log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
        out = logging.StreamHandler(sys.stdout)
        out.setLevel(logging.INFO)
        out.setFormatter(log_format)
        self.log.addHandler(out)

    def hparams_to_canonical_path(self, hparam_collection: HparamCollection
    ) -> os.PathLike:
        """Converts canonical_str from hparam_collection into path.

        Args:
            hparam_collection (HparamCollection): collection of Hparams.

        Returns:
            os.PathLike: canonical path beginning with self.hparams.job_out_dir
        """
        canonical_str = str(hparam_collection)
        canonical_str = canonical_str.replace(self.os.sep, '|')
        canonical_path = canonical_str.split(' ')
        return self.os.join(self.hparams.job_out_dir, *canonical_path)

    def list_checkpoints_by_iter(self, base_path: os.PathLike) -> typing.List[os.PathLike]:
        """Finds filepath of most recent checkpoint by highest epoch/iteration
        numbers in filename. Searches all subdirectories of given base_path recursively,
        relative to current working directory.
        Does not open checkpoint file to verify any iteration values.
        Simply sorts by the last 2 integers present in the filename,
        assuming they are epoch and iteration respectively.

        Args:
            base_path (str): root directory to search from.

        Returns:
            list[tuple[os.PathLike, int, int]: list of tuples of
                path to checkpoint file (including base_path),
                epoch, and iteration, sorted by ascending epoch/iter.
        """
        checkpoints = []
        for path, ckpt_name in self.os.find(base_path, suffix='.ckpt'):
            integers = [int(i) for i in re.findall(r'\d+', ckpt_name)]
            checkpoint_path = self.os.join(path, ckpt_name)
            if len(integers) < 2:
                self.log.warn(f'Listing checkpoints at {base_path}, ' +
                    f'checkpoint {checkpoint_path} epoch/iter missing?')
            else:
                epochs = integers[-2]
                iters = integers[-1]
                checkpoints.append((checkpoint_path, epochs, iters))
        return sorted(checkpoints, key=lambda tup: (tup[1], tup[2], tup[0]))

    def _replace_latest_ckpt_paths(self, hparams: dict) -> dict:
        """Replaces any `[base_path]/JOB_latest_checkpoint`
        with latest checkpoint in `sort_checkpoints([base_path])`.
        If latest checkpoint not available, remove the hparam.

        Args:
            base_path (str): root directory to search from.

        Returns:
            os.PathLike: path to most recent checkpoint file
        """
        new_hparams = {}
        for k, v in hparams.items():
            if type(v) is str and v.endswith(self.LATEST_CKPT_SHORTHAND):
                base_path = v[:-len(self.LATEST_CKPT_SHORTHAND)]
                checkpoints_by_iter = self.list_checkpoints_by_iter(base_path)
                # replace path if checkpoints available, else omit hparam
                if len(checkpoints_by_iter) > 0:
                    latest_ckpt_path, _, _ = checkpoints_by_iter[-1]
                    self.log.info(f'Replaced hparam `{k}` with {latest_ckpt_path}, ' +
                        f'original value was {v}')
                    new_hparams[k] = latest_ckpt_path
            else:  # copy other hparams
                new_hparams[k] = v
        return new_hparams

    def replicates(self, base_path: os.PathLike, include: set = None) -> typing.List[os.PathLike]:
        """Looks in `path` for replicates.
        Returns list of job script filenames for replicates in the category `status`.
        `canonical_path` may not exist, in which case an empty list is returned.

        Args:
            canonical_path (os.PathLike): base path to search for replicates.
            status (str): set of categories of replicates to include, one or more of
                {'empty', 'created', 'started', 'running', 'error', 'timeout', 'complete'}.
                Categories are mutually exclusive. If None return all replicates.
                'empty' - no script file
                'created' - script file exists but no stdout file
                'started' - stdout file exists but no run.py started notification
                'running' - run.py started but none of the notifications below
                'error' - error notification in stderr
                'timeout' - timeout notification in stderr
                'complete' - run.py complete notification in stdout

        Returns:
            tuple[list[int], list[os.PathLike]]: list of replicate numbers and list of paths to jobs
        """
        # create canonical_path if not existing
        replicates, filenames = [], []
        for f in self.os.list(base_path, suffix=self.os.sep):
            # only consider positive integers as replicates
            try:
                replicate_number = int(f[:-1])
            except ValueError:
                continue
            if replicate_number < 1:
                continue
            # filter replicates by status

            # job script doesn't exist
            status = 'empty'
            # job script exists but no output
            path = self.os.join(base_path, f)
            if self.os.type_of(self.os.join(path, self.SCRIPT_NAME)) == 'file':
                status = 'created'
                # stdout exists
                if self.os.type_of(self.os.join(path, self.STDOUT_FILENAME)) == 'file':
                    status = 'started'
                    if self._find_line(
                        self.os.join(path, self.STDOUT_FILENAME),
                        Runner.LOG_FORMAT_RE.format(LOG_level='INFO') + \
                            re.escape(Runner.START_MESSAGE)) \
                    is not None:
                        status = 'running'
                    # run.py has finished
                    if self._find_line(
                        self.os.join(path, self.STDOUT_FILENAME),
                        Runner.LOG_FORMAT_RE.format(LOG_level='INFO') + \
                            re.escape(Runner.END_MESSAGE)) \
                    is not None:
                        status = 'complete'
                    # stderr exists
                    if self.os.type_of(self.os.join(path, self.STDERR_FILENAME)) == 'file':
                        # run.py indicates exception
                        if self._find_line(
                            self.os.join(path, self.STDERR_FILENAME),
                            Runner.LOG_FORMAT_RE.format(LOG_level='CRITICAL') + \
                                re.escape(Runner.ERROR_MESSAGE)) \
                        is not None:
                            status = 'error'
                        # slurm timeout in stderr
                        # replace self.TIMEOUT_LAST_LINE to detect other message
                        if self._find_line(
                            self.os.join(path, self.STDERR_FILENAME),
                            self.TIMEOUT_LAST_LINE) \
                        is not None:
                            status = 'timeout'
            # filter replicate
            if include is None or status in include:
                replicates.append(replicate_number)
                filenames.append(f)
        return replicates, filenames

    def _find_line(self, filename: os.PathLike, pattern: str):
        for line in self.os.read(filename):
            matches = re.match(pattern, line)
            if matches is not None:
                return matches
        return None

    def new_replicate(self, base_path: os.PathLike) -> os.PathLike:
        """Checks if `base_path` exists.
        If non-existing, creates subdirectories as needed.
        Else, finds highest replicate number and
        creates/returns new subdirectory which increments the replicate number.

        Args:
            base_path (os.PathLike): base path to search for replicates.

        Returns:
            os.PathLike: path to new replicate subdirectory,
                replicate number guaranteed to be largest in base_path
        """
        # create base_path if not existing
        self.os.mkdir(base_path)
        replicates, _ = self.replicates(base_path)
        highest_replicate = 0
        if len(replicates) > 0:
            highest_replicate = sorted(replicates)[-1]
        # increment integer dirs, left pad with zeros
        next_replicate = str(highest_replicate + 1).zfill(self.N_REPLICATE_ZEROS)
        new_replicate_path = self.os.join(base_path, next_replicate)
        self.os.mkdir(new_replicate_path)
        return new_replicate_path

    def create(self,
        hparams: dict,
        replicate_path: os.PathLike = None,
        replace_existing = False
    ) -> os.PathLike:
        """Generates job script.
        Checks hparams using `get_parser` in run.py
        Defines `--default_root_dir` using canonical path.
        Defines other pytorch_lightning args from self.hparams.
        If `script_path` not defined,
        calls `hparams_to_canonical_str` to define output location and
        calls `new_replicate` to create job path.
        Calls `self._create` to create job script.

        Args:
            hparams (dict): hparams for run.py
            replicate_path (os.PathLike): if None, create script in new replicate dir,
                else create script in existing replicate_path. Defaults to None.
            replace_existing (str): if False, do not replace existing script file
                if exists in replicate_path, else overwrite. Defaults to False.

        Returns:
            os.PathLike: path to job script, or None if script not created.
        """
        # check hparams are valid in run.py and find canonical_path
        parser = Initializer.get_parser(hparams)
        hparams = Hparams.parse_dict(hparams, parser)
        hparams = Hparams.changed_hparams(hparams, parser)
        if replicate_path is None:
            canonical_path = self.hparams_to_canonical_path(hparams)
            replicate_path = self.new_replicate(canonical_path)
        # update hparams with job args, canonical_path, and latest ckpts
        job_params = Hparams.changed_hparams(self.hparams, self.default_hparams())
        hparams = {**hparams, **job_params,
            'default_root_dir': replicate_path}
        hparams = self._replace_latest_ckpt_paths(hparams)
        # create and save script
        script = self._create(hparams) + '\n' + \
            self.SCRIPT_LAST_LINE.format(
                JOB_author=self.hparams.job_author,
                JOB_class=type(self).__name__,
                JOB_version=VERSION,
                JOB_date=str(datetime.now()),
            ) + '\n'
        script_path = self.os.join(replicate_path, self.SCRIPT_NAME)
        if self.os.type_of(script_path) != 'none' and not replace_existing:
            self.log.warn(f'Script at {script_path} already exists, ' +
                'set `replace_existing = True` to overwrite.')
            return None
        self.os.write(script, script_path)
        return script_path

    def resume_latest(self, base_path: os.PathLike) -> os.PathLike:
        """Continues training from latest checkpoint available.

        Args:
            base_path (os.PathLike): path containing checkpoints.

        Returns:
            os.PathLike: path to job script, or None if script not created.
        """
        latest_ckpt_path = self.list_checkpoints_by_iter(base_path)[-1]
        hparams = torch.load(latest_ckpt_path)['hparams']
        hparams['resume_from_checkpoint'] = latest_ckpt_path
        return self.create(hparams)

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
