from seqmodel.abstract_hparam import Hparams
from seqmodel.abstract_hparam import HparamCollection


class DatasetHparams(Hparams):
    def _default_hparams():
        return {
            'batch_size': (16, 'bs', int,
                'number of samples in each training minibatch'),
        }

class SeqIntervalDatasetHparams(DatasetHparams):
    def _default_hparams():
        return {
            'seq_file': ('data/seq.fasta', 'seq', str,
                'path to sequence file'),
            'intervals': (None, 'int', str,
                'path to interval files for training split, use all sequences if None.'),
            'seq_len': (2000, 'len', int,
                'length of sampled sequence'),
            'skip_len': (None, 'skip', str,
                'how many characters to skip before next sample, ' + \
                                'defaults to seq_len'),
            'min_len': (None, 'mlen', str,
                'smallest interval length to sample from, ' + \
                                'defaults to seq_len'),
            'randomize_start_offsets': (True, 'randstart', bool,
                'move sampling start position by random number ' + \
                                'less than seq_len'),
            'drop_incomplete': (True, 'noinc', bool,
                'remove first and/or last samples with length ' + \
                                'less than seq_len, if False and start_offset > 0, ' + \
                                'this will pad the start of the first sample.'),
            'reverse_prop': (0.5, 'prev', float,
                'proportion of samples to reverse.'),
            'complement_prop': (0.5, 'pcompl', float,
                'proportion of samples to complement.'),
        }

class LinearDecoderHparams(Hparams):
    def _default_hparams():
        return {
            'decode_dims': (None, 'ddec', int,
                'number of dimensions in intermediate layers, ' + \
                                'if None set to 2*in_dims'),
            'n_decode_layers': (2, 'ndec', int,
                'number of linear layers'),
            'decode_dropout': (0., 'ddrop', float,
                'dropout between linear layers'),
        }

class PositionEncoderHparams(Hparams):
    def _default_hparams():
        return {
            'posencoder_dropout': (0., 'posdrop', float,
                'dropout after positional encoder'),
        }

class TransformerEncoderHparams(Hparams):
    def _default_hparams():
        return {
            'repr_dims': (512, 'd', int,
                'number of dimensions in representation layer'),
            'feedforward_dims': (None, 'dff', int,
                'number of dimensions in feedforward (fully connected) layer, ' +
                                'if None set to 2*repr_dims'),
            'n_heads': (4, 'nh', int,
                'number of attention heads'),
            'n_layers': (4, 'nl', int,
                'number of attention layers'),
            'dropout': (0., 'drop', float,
                'proportion between [0., 1.] of dropout to apply between module layers.'),
        }

class TaskHparams(Hparams):
    def _default_hparams():
        return {
            # batch
            'accumulate_grad_batches': (1, 'bacc', int,
                'average over this many batches before backprop (pytorch_lightning)'),
            # optimizer
            'lr': (3e-4, 'lr', float,
                'learning rate'),
            'adam_beta_1': (0.9, 'b1', float,
                'beta 1 parameter for Adam optimizer'),
            'adam_beta_2': (0.99, 'b2', float,
                'beta 2 parameter for Adam optimizer'),
            'adam_eps': (1e-6, 'eps', float,
                'epsilon parameter for Adam optimizer'),
            'weight_decay': (0.01, 'wd', float,
                'weight decay for Adam optimizer'),
            'gradient_clip_val': (10., 'clip', float,
                'limit max abs gradient value, ' + \
                                'no clipping if 0 (pytorch lightning)'),
        }

class PtMaskHparams(TaskHparams):
    def _default_hparams():
        return {
            'keep_prop': (0.01, 'pk', float,
                'proportion of sequence positions to apply identity loss.'),
            'mask_prop': (0.13, 'pm', float,
                'proportion of sequence positions to mask.'),
            'random_prop': (0.01, 'pr', float,
                'proportion of sequence positions to randomize.'),
        }

class MatFileDatasetHparams(DatasetHparams):
    def _default_hparams():
        return {
            'mat_file': ('data/train.mat', 'mat', str,
                'path to matlab file containing training data'),
        }

class FtDeepSeaHparams(TaskHparams):
    def _default_hparams():
        return {
            #TODO
        }

class InitializerHparams(Hparams):
    def _default_hparams():
        return {
            'init_version': (None, 'ver', str,
                'code version number, increment if hparam functionality changes'),
            'init_task': (None, 'tsk', str,
                '[ptmask, FtDeepSea] objects to load'),
            'init_mode': ('train', 'mode', str,
                '[train, test] whether to run training or inference'),
            'precision': (32, 'pre', int,
                '32 or 16 bit training (pytorch lightning)'),
            'load_encoder_from_checkpoint': (None, 'cpenc', str,
                'path to encoder checkpoint, replaces encoder from ' + \
                'load_from_checkpoint or resume_from_checkpoint'),
        }


class RunHparamCollection(HparamCollection):
    def _hparam_list():
        return [
            InitializerHparams,
            PositionEncoderHparams,
            TransformerEncoderHparams,
            LinearDecoderHparams,
        ]

class PtMaskHparamCollection(RunHparamCollection):
    def _hparam_list():
        return [
            SeqIntervalDatasetHparams,
            PtMaskHparams,
        ]

class FtDeepSeaHparamCollection(RunHparamCollection):
    def _hparam_list():
        return [
            MatFileDatasetHparams,
            FtDeepSeaHparams,
        ]

def get_hparam_collection(hparams: dict = None) -> HparamCollection:
    if hparams is None:
        parser = InitializerHparams.to_parser()
        args, _ = parser.parse_known_args()
        init_args = vars(args)
    else:
        init_args = InitializerHparams(**hparams)
    if init_args['init_version'] == '0.0.0' or \
            init_args['init_version'] == '0.0.1':
        if init_args['init_task'] == 'ptmask':
            return PtMaskHparamCollection
        elif init_args['init_task'] == 'FtDeepSea':
            return FtDeepSeaHparamCollection
        else:
            raise ValueError(f"Invalid model type {init_args['init_task']}.")
    else:
        raise ValueError(f"Unknown seqmodel version {init_args['init_version']}")
