TODO LIST
=========
- encapsulate selecting `HparamCollection` as separate function outside `run.py` called by `Job`.
    - document supported `pytorch_lightning` hparams.
- implement ssh jobs and Slurm jobs
    - add flag in jobs to submit job with '--fast_dev_run=True' to test, then without '--fast_dev_run=True' to run
    - implement immediate job status reporting after submit
    if no errors.
    - unit testing create jobs
    - document `os_interface.py`
- batch size search in job.py
- implement positional encoder
- implement ft tasks
- log: seq coverage, loss, accuracy by class, median auc, ptmask, accuracy by mask type?


SETUP
=====
Requires pytorch 1.9, cuda 10.2, for training with native 16-bit precision,
and pl_bolts implementation of learning rate warmup scheduler.


INFERENCE
=========

Default Behavior
----------------
- use virtualenv to set up python environment on device
- checkout code to on-device storage (localscratch)
- copy and/or extract data to on-device storage
- set `default_root_dir` to output to target storage device

Paths
-----
- working directory is assumed to be root of repository
- data is in `./data/[type]`, copy or symlink
- outputs default to `./out` unless `--default_root_dir` is set

Running
-------
- run `python seqmodel/run.py` with appropriate command line arguments.
- set `--mode=test` in `seqmodel.run` for inference


TRAINING
========

Automatic Jobs
--------------
Submitting individual jobs using python interface:
- import `seqmodel.job.Job` or a subclass in local or remote Jupyter notebook
- create `Job` object with appropriate credentials
- call `submit` with relevant hparams to create job
- `Job` will log submitted jobs to `job_out_dir`, and save all model outputs to the same path
- `Job` reports on replicate status using stdout and stderr redirected to files:
    - `created`: if job script exists but stdout and stderr missing
    - `error`: exception in stderr
    - `complete`: job complete message in stdout
    - `timeout`: task kill or Slurm timeout message in stdout
    - `running`: none of the above

Multiple Jobs
-------------
Creating multiple jobs for hparam tuning:
- create a regular `Job` object as described previously
- import `seqmodel.job.search` in local or remote Jupyter notebook
- describe hparams to vary in a list or dict format as required by functions in `search`
- run appropriate functions in `search`


DEVELOPMENT
===========


Repository
----------
- Primary branch is `main`.
- Keep all versions tagged on `main`.
- Use separate branches for code changes needed for specific experiments or debugging.
- One line per import to make diffs more readable.

Sequence Data
-------------
- make no assumptions about length or characters in sequence alphabet, instead pass `dataset.seq.Alphabet` objects
- use one single file per data type (e.g. FASTA, BED), concatenate multiple files together if needed
- all sequence names must be unique, specifically, the first part of the sequence descriptor split by whitespace should be unique (due to using pyfaidx to retrieve sequences)
- all coordinates are assumed zero-indexed, end-exclusive (same behaviour as python)
- sequence positions are identified using hash of full name/descriptor of sequence, and start coordinates
- hash of the full FASTA name/descriptor is used to avoid sending string type to GPU
- all metadata is in the form of a single tensor of type torch.long, of dimension `N, ...`  where `N` is batch size, and remaining is defined by dataset (e.g. for a sequence, `N, 2` where the 2 dimensions correspond to hash of name and start coordinate)
- all data tensors have dimensions `N, S, E` where `N` is batch size, `S` is sequence length, and `E` is number of dimensions
- a dataset should inherit from one of `[torch.utils.data.Dataset, torch.utils.data.IterableDataset]`, and one of `[seqmodel.dataset.UnsupervisedDataset, seqmodel.dataset.SupervisedDataset]`.
- dataloaders are reloaded every epoch by default


Hyperparameters
---------------
Any class requiring registered hyperparameters should create a container which subclasses of `seqmodel.hparam.Hparams` in `hparam.py`.
- this allows hparams to be imported as `seqmodel.hparam` without any other dependencies.
- hparams form a single namespace (no duplicate names or overriding allowed, each hparam is defined by only one class)
- hparams are inherited by subclasses
- class methods in `Hparams` allow finding default values, generating arg parser,
instance of `Hparams` stores actual hyperparameter values.
- store `Hparams` object in `self.hparams`
- when initializing `Hparams`, missing keys are replaced by default values
- only defaults can be `None`, initializing with `None` gives default value.
- To use a group of `Hparam` objects, create subclass of `HparamCollection`. Select collection in `run.py`.
- `Hparam` objects in `hparam.py` are used by `job.py` to construct canonical paths.

For unit testing:
- each hparam must have a non-empty help string

Versions
--------
Version number in `seqmodel.VERSION` indicates hyperparameter and model loading compatibility.
Development guidelines:
- tag versions in git, and update latest version tag with each commit
- increment build/minor version for any changes to hyperparameter or model loading behaviour
- increment major version number for compatibility breaking changes
- update `run.py` to maintain backwards compatibility when loading models, always check version number
- include test cases for loading current and older versions, including `load_checkpoint_path`, `resume_from_checkpoint`, and `load_encoder_from_checkpoint`.

Jobs
----
- `template_*.sh` files are used to define script templates for jobs
- `str.format` is called to replace arguments in `{}` braces in script templates
- if jobs are remote, scripts are copied to remote in the same location where outputs are stored
- jobs are named by default using the canonical string
- any repeat jobs increment replicate number by appending `_version_[n]` to the script name
- similarly, outputs of replicates are in subdirectories labelled `version_[n]`
