import sys
sys.path.append('.')
import logging
from seqmodel.init import Initializer
from seqmodel.hparam import get_hparam_collection
from seqmodel.util import find_subclasses

class Runner():

    START_MESSAGE = 'seqmodel/run.py started.'
    END_MESSAGE = 'seqmodel/run.py finished.'
    ERROR_MESSAGE = 'seqmodel/run.py terminated with ERROR.'
    LOG_FORMAT_RE = '\[(.+)\] \[{LOG_level}\] - '

    def __init__(self):
        self.hparams_to_init = {}
        for module in find_subclasses(Initializer,
                ['seqmodel/init.py'], exclude=[Initializer]):
            k = module.hparam_collection()
            self.hparams_to_init[k] = module

        log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
        # write everything to stdout
        out = logging.StreamHandler(sys.stdout)
        out.setLevel(logging.INFO)
        out.setFormatter(log_format)
        self.log.addHandler(out)

    def run(self):
        try:
            # indicate running
            self.log.info(self.START_MESSAGE)
            # get appropriate hparam collection
            hparam_collection_obj = get_hparam_collection()
            parser = hparam_collection_obj.to_parser()
            args = parser.parse_args()
            hparam_str = ['Hyperparameters:'] + \
                [f'{k} = {v}' for k, v in vars(args).items()]
            self.log.info('\n'.join(hparam_str))
            # initialize objects
            init_obj = self.hparams_to_init[hparam_collection_obj]
            runnable = init_obj(**vars(args))
            # run train/test loop
            runnable.run(args)
            # indicate successful completion
            self.log.info(self.END_MESSAGE)
            sys.exit(0)
        except Exception as e:
            # indicate error
            self.log.error('Run stopped on exception', exc_info=e)
            self.log.critical(self.ERROR_MESSAGE)
            sys.exit(1)


if __name__ == '__main__':
    Runner().run()
