from argparse import ArgumentParser
import pytorch_lightning as pl
from seqmodel import VERSION
from seqmodel.hparam import Hparams
from seqmodel.task import Task


class Initializer(Hparams):
    """Wrapper for hparams needed to initialize dataset/model/task.
    """
    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--model_version', default=VERSION, type=str,
                            help='code version number, increment if hparam functionality changes')
        parser.add_argument('--model_type', default=None, type=str,
                            help='[ptmask, ftdeepsea] objects to load')
        parser.add_argument('--load_encoder_from_checkpoint', default=None, type=str,
                            help='path to encoder checkpoint, replaces encoder from' +
                                ' --load_from_checkpoint or --resume_from_checkpoint')
        return parser

    @staticmethod
    def get_parser():
        """Get parser with default_hparams from all necessary objects for a
        particular configuration of the dataset/model/task defined by `--model_type`.
        Parses `Initializer` args first to understand which objects to include,
        then adds default_hparams of those objects to the parser.
        Also used to validate hparams in `seqmodel.job.Job`.

        Returns:
            ArgumentParser: parser for dataset/model/task hparams
        """
        parser = ArgumentParser()
        # add Initializer args
        # parse known args to get object structure
        # add object args
        parser = pl.Trainer.add_argparse_args(parser)
        return None #TODO

    @staticmethod
    def _initialize_objects(hparams: dict)-> Task:
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
    def initialize_objects(parser: ArgumentParser)-> Task:
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


def train(task: Task):
    """Runs training loop on task.
    """
    pass

if __name__ == '__main__':
    # initialize objects
    # run train/test loop
    pass #TODO
