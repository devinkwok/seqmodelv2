from seqmodel.task import Task


class Finetune(Task):

    @staticmethod
    def _default_hparams(parser):
        parser.add_argument('--sum_representation', default=True, type=bool,
                            help='if True, use mean of representation vectors' +
                                ' over all sequence positions for classification,' +
                                ' else use representation from first position.')
        return parser

    def __init__(self):
        pass

    def forward(self):
        pass

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def validation_epoch_end(self):
        pass
