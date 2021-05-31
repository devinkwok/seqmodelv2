import typing
from seqmodel.job import Job


def grid_search(job_base: Job, hparam_base: dict, grid_hparams: typing.Dict[str, list]):
    """Function for hparam optimization via grid search.
    For $k$ lists of hparam values of length $n_1, n_2, \dots, n_k$,
    creates $n_1 \times n_2 \times \dots \times n_k$ jobs.

    Args:
        job_base (Job): object to create jobs from
        hparam_base (dict): hparams that are consistent across all jobs
        grid_hparams (typing.Dict[str, list]): hparams to create grid over, where
            each key is one axis of the grid and each list element is a coordinate in the grid.
    """
    pass #TODO


def multi_job_submit(job_base: Job, hparam_base: dict, hparam_changes: typing.List[dict]):
    """Function for submitting multiple jobs with arbitrary hparams.

    Args:
        job_base (Job): object to create jobs from
        hparam_base (dict): hparams that are consistent across all jobs
        hparam_changes (typing.List[dict]): each list item defines
            a separate job with hparams to replace hparam_base
    """
    pass #TODO
