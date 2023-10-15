# A Lightweight Method for Modeling Confidence in Recommendations with Learned Beta Distributions
This repository contains the code used for the experiments in "A Lightweight Method for Modeling Confidence in Recommendations with Learned Beta Distributions" published at RecSys 2023 ([open access article](https://dl.acm.org/doi/abs/10.1145/3604915.3608788)).

Citation
--------

If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our RecSys 2023 paper:
```
@inproceedings{knyazev2023alightweight,
  Author = {Knyazev, Norman and Oosterhuis, Harrie},
  Booktitle = {Seventeenth ACM Conference on Recommender Systems (RecSys '23)},
  Organization = {ACM},
  Title = {A Lightweight Method for Modeling Confidence in Recommendations with Learned Beta Distributions},
  Year = {2023}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.

Usage
-------

This code makes use of [Python 3](https://www.python.org/) and the following packages: [jupyter](https://jupyter.org), [matplotlib](https://matplotlib.org), [numpy](https://numpy.org/), [scipy](https://scipy.org), [pandas](https://pandas.pydata.org), [tqdm](https://tqdm.github.io), [dotenv](https://pypi.org/project/python-dotenv/), [tensorflow==2.12.0](https://tensorflow.org) and [tensorflow-probability](https://www.tensorflow.org/probability). Make sure they are installed.

The code can be accessed by running `jupyter notebook .` in the project folder and navigating to `src/notebooks`.

The process to replicate the results reported in the publication consists of four steps:
- Modify the variable `PROJECT_ROOT` in the `.env` file contained in the root directory of this project to point to the global path of the root directory.
- Run `src/notebooks/preprocess_data.ipynb` to download and preprocess the dataset used for evaluation.
- Run `src/notebooks/run_models.ipynb` to train the models and export test fold predictions. Each cell trains one model on every one of 10 train-test splits and for each run exports the test set predictions (and the intermediate representations) to `logs/LBD_results/{model_name}/{model_name}-{fold_id}-0/export`. 
- Run `src/notebooks/RQ{research_question_number}` to load the above predictions and to obtain the reported numerical results and/or visualizations.

Useful tips:
- By default, training and evaluating different models on multiple folds in `src/notebooks/run_models.ipynb` is done in a sequential manner. It is also possible to train only some of the models within each runtime by running the chosen cells. Alternatively, all runs for one model can be executed in parallel by setting `JOB_TYPE="new_process"` or via slurm by setting `JOB_TYPE="slurm"`. For the latter, ensure that `src/modules/utils/slurm/slurm_header.txt` corresponds to your slurm environment.
- To evaluate a model on a subset of test folds (e.g. 1), the folds can be specified in the model's config under `data_params['params']['folds_to_use_outer']`, for example `[0, 2, 3, 9]`.
- A single training-evaluation loop can also be executed by running the function `src.modules.training.train_run` with appropriate parameters.
