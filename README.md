SETUP
=====
#TODO


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

Primary branch is `main`. Keep all versions tagged on `main`. Use separate branches for code changes needed for specific experiments or debugging.

Sequence Data
-------------
- make no assumptions about length or characters in sequence alphabet, instead pass `dataset.seq.Alphabet` objects.
- sequence data can be split between multiple files, but each name must be unique

Hyperparameters
---------------
Any class registering a hyperparameter is a subclass of `seqmodel.hparam.Hparams`
- hparams form a single namespace (no duplicate names or overriding allowed, each hparam is defined by only one class)
- hparams are inherited by subclasses
- default hparams are stored in an `ArgumentParser` object
- default_hparams are required arguments for object initialization, although other arguments can be provided. When calling `__init__` on a subclass of `Hparams`, all defined hparams are stored in `self.hparams`, and the remaining arguments are ignored.
- missing keys in hparams are replaced by default values
- `run.py` combines all parsers needed to define the model objects. `job.py` has parsers which are independent of `run.py`.
- static methods in `hparam` require an `ArgumentParser` parameter, this allows parsers from multiple objects to be combined. When operating on one object only, call that object's `default_hparams()` to get a parser.
- use `hparam.parse_dict()` to convert between `ArgumentParser` and `dict`
- use type `hparam.str2bool` instead of `bool` for booleans, this solves a known argparse bug

For unit testing:
- subclasses of `Hparams` must be at the top level of each module
    (i.e. defined by `class Name(Hparams)` in each `.py` file, not defined within another class)
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
