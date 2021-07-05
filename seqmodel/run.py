import sys
sys.path.append('.')
import logging
from datetime import datetime
import torch.nn as nn
import pytorch_lightning as pl
from argparse import ArgumentParser
from seqmodel import Hparams
from seqmodel.dataset import DnaAlphabet
from seqmodel.dataset import SeqIntervalDataset
from seqmodel.model import LinearDecoder
from seqmodel.model import PositionEncoder
from seqmodel.model import TransformerEncoder
from seqmodel.task import Task
from seqmodel.task import PtMask
from seqmodel.task import MatFileDataset
from seqmodel.task import FtDeepSEA


class Initializer(Hparams):
    """Wrapper for hparams needed to initialize dataset/model/task.
    """
    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--init_version', default=None, type=str,
                            help='code version number, increment if hparam functionality changes')
        parser.add_argument('--init_task', default=None, type=str,
                            help='[ptmask, ftdeepsea] objects to load')
        parser.add_argument('--init_mode', default='train', type=str,
                            help='[train, test] whether to run training or inference')
        parser.add_argument('--load_encoder_from_checkpoint', default=None, type=str,
                            help='path to encoder checkpoint, replaces encoder from ' +
                                '--load_from_checkpoint or --resume_from_checkpoint')
        parser.add_argument('--precision', default=16, type=int,
                            help='32 or 16 bit training (pytorch lightning)')
        return parser

    @staticmethod
    def get_parser(args: dict = None):
        """Get parser with default_hparams from all necessary objects for a
        particular configuration of the dataset/model/task defined by `--init_task`.
        Parses `Initializer` args first to understand which objects to include,
        then adds default_hparams of those objects to the parser.
        Also used to validate hparams in `seqmodel.job.Job`.

        Args:
            args (dict): args to parse with initializer, if None parses command line input.

        Returns:
            ArgumentParser: parser for dataset/model/task hparams
        """
        parser = ArgumentParser()
        # parse Initializer args to determine which objects are needed
        parser = Initializer.default_hparams(parser)
        if args is None:  # get args from system
            init_args = vars(parser.parse_known_args())
        else:  # use supplied args from dict
            init_args = parse_known_dict(args, parser)

        # model objects
        parser = PositionEncoder.default_hparams(parser)
        parser = TransformerEncoder.default_hparams(parser)
        parser = LinearDecoder.default_hparams(parser)

        # data and task objects
        if init_args['init_task'] == 'ptmask':
            parser = StridedSeqSampler.default_hparams(parser)
            parser = PtMask.default_hparams(parser)
        elif init_args['init_task'] == 'ftdeepsea':
            parser = MatFileDataset.default_hparams(parser)
            parser = FtDeepSEA.default_hparams(parser)
        else:
            raise ValueError(f"Invalid model type {init_args['init_task']}.")

        # version specific behaviour
        if init_args['init_version'] == '0.0.0' or \
                init_args['init_version'] == '0.0.1':
            pass  # no version specific behaviour yet
        else:
            raise ValueError(f"Unknown seqmodel version {init_args['init_version']}")

        return parser

    @staticmethod
    def initialize_objects(self, args: dict = None) -> Task:
        """Initializes objects from parser (calls `_initialize_objects`).

        Args:
            parser (ArgumentParser): parser with hparams

        Returns:
            Task: object subclassing `Task`
        """
        parser = self.get_parser(args)
        parser = Initializer.default_hparams(parser)
        if args is None:
            hparams = vars(parser.parse_known_args())
        else:  # use supplied args from dict
            hparams = Hparams.parse_known_dict(args, parser)
        # common objects/hparams
        alphabet = DnaAlphabet()
        repr_dims = hparams['repr_dims']

        # model objects
        pos_encoder = PositionEncoder(len(alphabet), repr_dims, **hparams)
        encoder = TransformerEncoder(pos_encoder,
                    ActivationFn=nn.GELU, DropoutFn=nn.Dropout,
                    LayerNormFn=nn.LayerNorm, **hparams)

        # data and task objects
        if hparams['init_task'] == 'ptmask':
            dataset = StridedSeqSampler(**hparams)
            decoder = LinearDecoder(
                repr_dims,
                dataset.alphabet.n_char,
                **hparams)
            task = PtMask(dataset, encoder, decoder, **hparams)
        elif hparams['init_task'] == 'ftdeepsea':
            dataset = MatFileDataset(**hparams)
            decoder = LinearDecoder(
                repr_dims,
                dataset.target_dims,
                **hparams)
            task = FtDeepSEA(dataset, encoder, decoder, **hparams)
        else:
            raise ValueError(f"Invalid model type {hparams['init_task']}.")

        # version specific behaviour
        if hparams['init_version'] == '0.0.0' or \
                hparams['init_version'] == '0.0.1':
            pass  # no version specific behaviour yet
        else:
            raise ValueError(f"Unknown seqmodel version {hparams['init_version']}")

        return task

    @staticmethod
    def train(task: Task):
        pl.seed_everything(0)
        trainer = pl.Trainer(**hparams)
        if hparams['init_mode'] == 'train':
            trainer.fit(task)
        elif hparams['init_mode'] == 'test':
            trainer.test(task)
        args.callbacks


class Runner():

    START_MESSAGE = 'seqmodel/run.py started.'
    END_MESSAGE = 'seqmodel/run.py finished.'
    ERROR_MESSAGE = 'seqmodel/run.py terminated with ERROR.'
    LOG_FORMAT_RE = '\[(.+)\] \[{LOG_level}\] - '

    def run(self, hparams: dict = None):
        log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
        # writing to stdout
        out = logging.StreamHandler(sys.stdout)
        out.setLevel(logging.INFO)
        out.setFormatter(log_format)
        self.log.addHandler(out)
        # writing to stderr
        err = logging.StreamHandler(sys.stderr)
        err.setLevel(logging.ERROR)
        err.setFormatter(log_format)
        self.log.addHandler(err)

        try:
            # indicate running
            self.log.info(self.START_MESSAGE)
            # initialize objects
            task = Initializer.initialize_objects(hparams)
            self.trainer.train(task)
            # run train/test loop

            #TODO testing
            import time
            time.sleep(30)
            #TODO testing exception handling
            if datetime.now() is None:
                raise ValueError('TEST exception')

            # indicate successful completion
            self.log.info(self.END_MESSAGE)
            sys.exit(0)
        except Exception as e:
            # indicate error
            self.log(e)
            self.log.critical(self.ERROR_MESSAGE)
            sys.exit(1)


if __name__ == '__main__':
    Runner().run()
