# Slurm utilities
import copy
import glob
import json
import os
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.append(PROJECT_ROOT)

import src.modules.utils.utils as utils
import src.modules.data.postprocessing.utils as postprocessing_utils

PYTHON_HEADER = "#!/bin/sh\n"
SLURM_HEADER_RELATIVE_PATH = "src/modules/utils/slurm/slurm_header.txt"


def submit_new_process_job(job_dir=None, config=None, **kwargs):
    slurm_job_dir = os.path.join(PROJECT_ROOT, "src", "modules", "utils", "slurm") if job_dir is None else job_dir
    slurm_job_path = os.path.join(slurm_job_dir, "slurm_job.sh")
    slurm_job_command = f"python3 {os.path.join(PROJECT_ROOT, 'src', 'modules', 'utils', 'slurm', 'slurm_job.py')} "
    slurm_job_command += " ".join([f"--{k} {v}" for k, v in kwargs.items()])
    slurm_job_content = PYTHON_HEADER + slurm_job_command
    with open(slurm_job_path, "w") as output_file:
        output_file.write(slurm_job_content)
        print("Starting job in new process:")
        print(slurm_job_content)
    subprocess.run(["chmod", "+x", slurm_job_path])
    subprocess.Popen(slurm_job_path)

def submit_slurm_job(job_dir=None, config=None, **kwargs):
    slurm_job_dir = os.path.join(PROJECT_ROOT, "src", "modules", "utils", "slurm") if job_dir is None else job_dir
    slurm_job_path = os.path.join(slurm_job_dir, "slurm_job.sh")
    slurm_job_command = f"python3 {os.path.join(PROJECT_ROOT, 'src', 'modules', 'utils', 'slurm', 'slurm_job.py')} "
    slurm_job_command += " ".join([f"--{k} {v}" for k, v in kwargs.items()])
    slurm_job_content = SLURM_HEADER + slurm_job_command
    with open(slurm_job_path, "w") as output_file:
        output_file.write(slurm_job_content)
        print("Submitted slurm job: ")
        print(slurm_job_content)
    subprocess.run(["chmod", "+x", slurm_job_path])
    subprocess.run(["pwd"])
    subprocess.Popen(["sbatch", slurm_job_path])

def submit_same_process_job(config, job_dir=None, **kwargs):
    import src.modules.utils.slurm.slurm_job as slurm_job
    print("CONTINUING JOB INSIDE THE SAME PROCESS")
    slurm_job.main(run_config=config)

def query_slurm_job_state(job_id):
    slurm_cmd = f"sacct -j {job_id}"
    p = subprocess.Popen(slurm_cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = p.stdout.read().decode()
    header = output.splitlines()[0]
    datarow = output.splitlines()[2]
    dict_output = {k: v for k, v in zip(header.split(), datarow.split())}
    return dict_output["State"]


def preprocess_run_config(config, *args):
    _config = copy.deepcopy(config)
    print("Config preprocessing:")
    print(_config, args)
    for k, v in args:
        skip = False
        target = _config
        for _k in k[:-1]:
            if _k in target:
                target = target[_k]
            else:
                print(f"{_k} not in config at path config[{k}]. Ignoring.")
                skip = True
                break
        if not skip:
            target[k[-1]] = v
    return _config

def get_data_paths(split_outer, split_inner, sweep_run_config, project_root=PROJECT_ROOT):
    params = sweep_run_config["data_params"]["params"]

    data_root = postprocessing_utils.get_postprocessed_data_root(
        project_root, **params
    )

    if params["format"] != "csv":
        raise NotImplementedError(f"Input format: {params['format']} not implemented.")

    outer_split_root = os.path.join(data_root, f"outer={str(split_outer)}")
    inner_split_root = os.path.join(outer_split_root, f"inner={str(split_inner)}")

    paths = {
        "train": os.path.join(inner_split_root, "train", f"*.{params['format']}"),
        "validation": os.path.join(inner_split_root, "validation", f"*.{params['format']}"),
        "test": os.path.join(outer_split_root, "test", f"*.{params['format']}")
    }
    if "extra" in sweep_run_config["data_params"]:
        for k, v in sweep_run_config["data_params"]["extra"].items():
            paths[k] = os.path.join(inner_split_root, v)
            print(f"Added {k}:{v} to data paths.")

    output = {}
    for k, p in paths.items():
        if not len(glob.glob(p)):
            print(f"{k} data at {p} does not exist. Path not included.")
        else:
            output[k] = p
    return output

def postprocess_run_config(config, mapping):
    for i, x in enumerate(mapping):
        print(f"{i}: {x}")
    for target_map, source_map in mapping:
        target_entry, source_entry = config, config
        for k in target_map[:-1]:
            target_entry = target_entry[k]
        for k in source_map:
            source_entry = source_entry[k]
        target_entry[target_map[-1]] = source_entry
    return config


def slurm_sweep_run(sweep_run_config=None, sweep_run_name=None, sweep_id=None, job_type="same_process"):

    job_types = {
        "slurm": submit_slurm_job,
        "new_process": submit_new_process_job,
        "same_process": submit_same_process_job
    }

    log_root = os.path.join(PROJECT_ROOT, sweep_run_config["meta_params"]["log_root"])

    sweep_run_log_path = os.path.join(log_root, sweep_id, sweep_run_name)
    sweep_run_config_path = os.path.join(sweep_run_log_path, "config.json")

    os.makedirs(sweep_run_log_path, exist_ok=True)
    with open(sweep_run_config_path, "w") as output_file:
        json.dump(dict(sweep_run_config), output_file, indent=4)
        print(f"Wrote sweep run config to {sweep_run_config_path}.")

    data_params = sweep_run_config["data_params"]

    # Check which of the inner and outer folds need to be run: either all or `folds_to_use_*` if specified
    if "folds_to_use_outer" in data_params["params"] and data_params["params"]["folds_to_use_outer"] is not None:
        outer_splits = data_params["params"]["folds_to_use_outer"]
    else:
        outer_splits = list(range(data_params["params"]["num_splits_outer"]))
    if "folds_to_use_inner" in data_params["params"] and data_params["params"]["folds_to_use_inner"] is not None:
        inner_splits = data_params["params"]["folds_to_use_inner"]
    else:
        inner_splits = list(range(data_params["params"]["num_splits_inner"]))

    run_names = {i: {j: f"{sweep_run_name}-{i}-{j}" for j in inner_splits} for i in outer_splits}
    run_log_paths = {i: {j: os.path.join(sweep_run_log_path, run_names[i][j]) for j in inner_splits}
                     for i in outer_splits}

    # Submit slurm jobs
    for i in outer_splits:
        for j in inner_splits:
            run_log_path = run_log_paths[i][j]
            run_config_path = os.path.join(run_log_path, "config.json")
            data_paths = get_data_paths(i, j, sweep_run_config)
            params_to_adjust = [
                (("data_params", "params", "split_outer"), i),
                (("data_params", "params", "split_inner"), j),
                (("meta_params", "sweep_id"), sweep_id),
                (("meta_params", "run_name"), run_names[i][j]),
                (("meta_params", "sweep_run_name"), sweep_run_name),
                (("meta_params", "run_log_path"), run_log_paths[i][j]),
                (("data_params", "train", "data_path"), data_paths["train"]),
            ]
            # Validation and test sets can also be not provided
            for k in ["validation", "test"]:
                if k in data_paths:
                    params_to_adjust.append((("data_params", k, "data_path"), data_paths[k]))
            for k, path in data_paths.items():
                if k not in ["train", "validation", "test"]:
                    params_to_adjust.append((("data_params", "extra", k), data_paths[k]))

            _config = preprocess_run_config(dict(sweep_run_config), *params_to_adjust)
            if "param_remapping" in _config:
                _config = postprocess_run_config(_config, _config["param_remapping"])
            os.makedirs(run_log_path, exist_ok=True)
            with open(run_config_path, "w") as output_file:
                json.dump(dict(_config), output_file, indent=4)
                print(f"Wrote single run config to {run_config_path}.")
            job_fn = job_types[job_type]
            job_fn(job_dir=run_log_path, run_config_path=run_config_path, config=_config)
            time.sleep(1)


slurm_header_path = os.path.join(PROJECT_ROOT, SLURM_HEADER_RELATIVE_PATH)
with open(slurm_header_path) as input_file:
    SLURM_HEADER = input_file.read()
