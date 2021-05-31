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
- `Job` will log submitted jobs to log_path, and save all model outputs to the same path

Multiple Jobs
-------------
Creating multiple jobs for hparam tuning:
- create a regular `Job` object as described previously
- import `seqmodel.job.search` in local or remote Jupyter notebook
- describe hparams to vary in a list or dict format as required by functions in `search`
- run appropriate functions in `search`


DEVELOPMENT
===========

Sequence Data
-------------
- make no assumptions about length or characters in sequence alphabet, instead pass `dataset.seq.Alphabet` objects.
- sequence data can be split between multiple files, but each name must be unique

Hyperparameters
---------------
Any class registering a hyperparameter is a subclass of `seqmodel.hparam.Hparam`
- hparams form a single namespace (no duplicate names or overriding allowed)
- hparams are stored in a `dict`
- default hparams are stored in an `ArgumentParser` object
- missing keys in hparams are replaced by default values
- hparams are required arguments for object initialization, and stored in `self.hparams`
- all parsed hparams are available to all objects, but it is possible to not parse some hparams, such as in `job.py`, in particular `run.py` parses all hparams for model tasks
- use `Hparam.parse_dict()` to convert 

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
