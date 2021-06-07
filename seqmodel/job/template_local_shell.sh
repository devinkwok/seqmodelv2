{JOB_local_env_activate}
## run job
{JOB_commands} > {JOB_stdout_file} 2> {JOB_stderr_file}

## clean up by stopping virtualenv
deactivate
