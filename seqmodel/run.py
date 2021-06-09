import sys
sys.path.append('.')
from argparse import ArgumentParser
from seqmodel import hparam
from seqmodel.dataset.sampler import StridedSeqSampler
from seqmodel.model.decoder import LinearDecoder
from seqmodel.model.transformer import PositionEncoder, TransformerEncoder
from seqmodel.task import Task
from seqmodel.task.pt_mask import PtMask
from seqmodel.task.ft_deepsea import MatFileDataset, FtDeepSEA


class Initializer(hparam.Hparams):
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
            init_args = hparam.parse_known_dict(args, parser)

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
    def _initialize_objects(hparams: dict) -> Task:
        """Initializes objects from hparams

        Args:
            hparams (dict): parsed hyperparameters

        Returns:
            Task: object subclassing `Task`
        """
        dataset = None #TODO
        encoder = None #TODO
        decoder = None #TODO
        task = None #TODO
        return task

    @staticmethod
    def initialize_objects(parser: ArgumentParser) -> Task:
        """Initializes objects from parser (calls `_initialize_objects`).

        Args:
            parser (ArgumentParser): parser with hparams

        Returns:
            Task: object subclassing `Task`
        """
        return None #TODO

        @staticmethod
        def load_encoder_from_checkpoint(ckpt_path: str):
            return None #TODO

class PlTrainer(hparam.Hparams):

    @staticmethod
    def _default_hparams(parser):
        return parser

    @staticmethod
    def train(task: Task):
        """Runs training loop on task.
        """
        pass

if __name__ == '__main__':
    # initialize objects
    # run train/test loop
    print('TEST') #TODO
    raise ValueError('TEST exception')
