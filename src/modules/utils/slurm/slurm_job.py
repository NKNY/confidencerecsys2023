import argparse
import json
import os
import sys

import numpy as np

from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.append(PROJECT_ROOT)

import src.modules.training.training as training

def main(run_config):

    run_name = run_config["meta_params"]["run_name"]
    sweep_id = run_config["meta_params"]["sweep_id"]
    sweep_run_name = run_config["meta_params"]["sweep_run_name"]

    print(f"------------\nINITIATING JOB:\n------------\nSWEEP_ID: {sweep_id}\nTYPE: {sweep_run_name}, NAME: {run_name}, CONFIG: ")
    from pprint import pprint
    pprint(run_config)
    print(f"------------\nINITIATION COMPLETE\n------------\n")

    # Build the whole training process from the config
    training.train_run(**run_config)
