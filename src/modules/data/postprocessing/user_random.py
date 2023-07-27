import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import sklearn.model_selection

from typing import Literal

from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.append(PROJECT_ROOT)

import src.modules.data.postprocessing.utils as postprocessing_utils


# Need to be mindful of which users/items have enough examples
# Splitting on user basis
def load_data(data_path, columns=["uid", "iid", "rating"]):
    data = np.load(data_path)
    df = pd.DataFrame(data, columns=columns)
    return df


def load_metadata(metadata_path):
    with open(metadata_path, "r") as input_path:
        metadata = json.load(input_path)
    return metadata


def get_xval_folds(data, num_folds, seed=None, columns=["uid", "iid", "rating"]):
    skf = sklearn.model_selection.StratifiedKFold(num_folds, shuffle=True, random_state=seed)
    folds_idx = skf.split(data.index, data[columns[0]])
    # The above steps may take quite a while for bigger dataset.
    for train_idx, test_idx in folds_idx:
        train_df, test_df = data.loc[data.index[train_idx]].copy(), data.loc[data.index[test_idx]].copy()
        test_df, train_df = prune_on_columns(test_df, train_df, columns=columns[:2])
        yield train_df, test_df


def prune_on_columns(df1, df2, columns=["uid", "iid"]):
    values_to_prune = {col: get_column_set_difference(df1, df2, col) for col in columns}
    while any(list(values_to_prune.values())):
        idx_to_prune = np.zeros(len(df1), dtype=bool)
        for col, values in values_to_prune.items():
            idx_to_prune = idx_to_prune | df1[col].isin(values)
        df1 = df1.loc[~idx_to_prune]
        values_to_prune = {col: get_column_set_difference(df1, df2, col) for col in columns}
    return df1, df2


def get_column_set_difference(df1, df2, column):
    set_df1, set_df2 = set(df1[column].unique()), set(df2[column].unique())
    return set_df1 - set_df2


def split_train_test(data, test_pc, seed, columns=["uid", "iid", "rating"]):
    rng = np.random.default_rng(seed)
    test_df = data.groupby(columns[0]).sample(frac=test_pc, random_state=rng)
    test_idx = set(test_df.index)
    train_idx = list(set(data.index) - set(test_idx))
    train_df = data.copy(deep=True).loc[train_idx]

    test_df, train_df = prune_on_columns(test_df, train_df, columns=columns[:2])
    return train_df, test_df

def split_train_test_by_idx(data, train_idx, test_idx):
    train_df = data.copy(deep=True).loc[train_idx]
    test_df = data.copy(deep=True).loc[test_idx]
    return train_df, test_df


def save_split(train_data, test_data, parent_dir, split_idx, output_formats, split_prefix: Literal["inner", "outer"],
               num_shards=1, names=["train", "test"]):
    split_path = os.path.join(parent_dir, f"{split_prefix}={str(split_idx)}")
    postprocessing_utils.save(train_data, split_path, names[0], num_shards=num_shards, formats=output_formats)
    if len(test_data) > 0:
        postprocessing_utils.save(test_data, split_path, names[1], num_shards=num_shards, formats=output_formats)
    else:
        print(f"{names[1]} is empty. File not saved.")
    return split_path


def main(project_root, dataset, xval_outer, num_splits_outer, test_pc, xval_inner, num_splits_inner, validation_pc,
         global_seed, seeds_path, num_shards, output_formats=["csv, tfrecords"], idx_paths=None):
    # Note that if idx is not None, it assumes that idx for all subsets are based on the original train+val+test array and
    # that num_splits_inner/outer is 1.
    
    print("project_root", project_root)
    print("dataset", dataset)
    print("xval_outer", xval_outer)
    print("xval_inner", xval_inner)
    print("num_splits_outer", num_splits_outer)

    # Paths
    input_root = os.path.join(project_root, "data", "preprocessed", dataset, "user_random")
    data_path, metadata_path = os.path.join(input_root, "data.npy"), os.path.join(input_root, "metadata.json")
    output_root = postprocessing_utils.get_postprocessed_data_root(project_root, dataset, xval_outer, num_splits_outer,
                                                                   test_pc, xval_inner, num_splits_inner,
                                                                   validation_pc, strategy="user_random")

    # Confirm that clear how many splits and how much data in each split
    try:
        assert (xval_outer & (num_splits_outer > 1) & (test_pc == 1)) | (not xval_outer & (test_pc < 1))
        assert (xval_inner & (num_splits_inner > 1) & (validation_pc == 1)) | (not xval_inner & (validation_pc < 1))
    except AssertionError:
        raise AssertionError("For both outer&inner folds, (xval -> num_splits>1 and test_pc=1) and !xval -> test_pc<1)")
    
    # Confirm that if we're providing idx then it contains all relevant keys and num_splits is 1.
    try:
        assert (idx_paths is None) or ((isinstance(idx_paths, dict)) and (set(idx_paths.keys()) == {"train", "validation", "test"}) and (num_splits_outer == num_splits_inner == 1))
    except AssertionError:
        raise AssertionError("If idx provided it has to be a dict with all train/validation/test keys and num_splits is 1")
    
    # Calculate #random seeds for data splitting and how big is validation_pc when adjusted for size of training set.
    num_seeds_outer, num_seeds_inner = 1, num_splits_outer  # xval from 1 seed, repeated samples 1 per seed
    _validation_pc = validation_pc
    if not xval_outer:
        num_seeds_outer *= num_splits_outer

    # validation percentage seems to be frequently indicated as percentage of the train set
    # if not xval_inner:
    #     num_seeds_inner *= num_splits_inner
    #     denom = (1 - test_pc) if not xval_outer else (num_splits_outer - 1) / num_splits_outer
    #     _validation_pc /= denom  # Validation % adjusted to reflect that it reflects a bigger % of non-test set.

    # Sample seeds or load from dict
    if seeds_path is None:
        # Flattened list of inner and outer seeds
        _seeds_outer = postprocessing_utils.generate_random_seeds(num_seeds_outer, global_seed=global_seed)
        _seeds_inner = postprocessing_utils.generate_random_seeds(num_seeds_inner, global_seed=global_seed)
    else:
        with open(seeds_path, "r") as input_file:
            seeds = json.read(input_file)
            global_seed, _seeds_outer, _seeds_inner = seeds["global"], seeds["outer"], seeds["inner"]

    # Generators (easier to work with)
    seeds_outer = (x for x in _seeds_outer)
    seeds_inner = (x for x in _seeds_inner)

    # Load data and metadata
    data, metadata = load_data(data_path), load_metadata(metadata_path)

    # Save seeds
    os.makedirs(output_root, exist_ok=True)

    if xval_outer:
        seed_outer = next(seeds_outer)
        outer_folds = get_xval_folds(data, num_splits_outer, seed=seed_outer)
            
        for i, (_train_data, test_data) in enumerate(outer_folds):
            outer_split_path = save_split(_train_data, test_data, output_root, i, output_formats, "outer",
                                          num_shards, names=["train", "test"])
            if xval_inner:
                seed_inner = next(seeds_inner)
                inner_folds = get_xval_folds(_train_data, num_splits_inner, seed=seed_inner)
                for j, (train_data, validation_data) in enumerate(inner_folds):
                    save_split(train_data, validation_data, outer_split_path, j, output_formats, "inner",
                               num_shards, names=["train", "validation"])
            else:
                for j, seed_inner in zip(range(num_splits_inner), seeds_inner):
                    train_data, validation_data = split_train_test(_train_data, _validation_pc, seed=seed_inner)
                    save_split(train_data, validation_data, outer_split_path, j, output_formats, "inner",
                               num_shards, names=["train", "validation"])

    elif not xval_outer:
        for i, seed_outer in zip(range(num_splits_outer), seeds_outer):
            if idx_paths is None:
                _train_data, test_data = split_train_test(data, test_pc, seed=seed_outer)
            else:
                idx_train, idx_test = np.load(idx_paths["train"]), np.load(idx_paths["test"])
                _train_data, test_data = split_train_test_by_idx(data, idx_train, idx_test)
            outer_split_path = save_split(_train_data, test_data, output_root, i, output_formats, "outer",
                                          num_shards, names=["train", "test"])
            if xval_inner:
                seed_inner = next(seeds_inner)
                inner_folds = get_xval_folds(_train_data, num_splits_inner, seed=seed_inner)
                for j, (train_data, validation_data) in enumerate(inner_folds):
                    save_split(train_data, validation_data, outer_split_path, j, output_formats, "inner",
                               num_shards, names=["train", "validation"])
            else:
                for j, seed_inner in zip(range(num_splits_inner), seeds_inner):
                    if idx_paths is None:
                        train_data, validation_data = split_train_test(_train_data, _validation_pc, seed=seed_inner)
                    else:
                        idx_train, idx_validation = np.load(idx_paths["train"]), np.load(idx_paths["validation"])
                        train_data, validation_data = split_train_test_by_idx(data, idx_train, idx_validation)
                    save_split(train_data, validation_data, outer_split_path, j, output_formats, "inner",
                               num_shards, names=["train", "validation"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", action="store", required=True)
    parser.add_argument("--xval_outer", action="store_true", default=False)
    parser.add_argument("--xval_inner", action="store_true", default=False)
    parser.add_argument("--num_splits_outer", dest="num_splits_outer", action="store", type=int, default=1)
    parser.add_argument("--num_splits_inner", dest="num_splits_inner", action="store", type=int, default=1)
    parser.add_argument("--test_pc", action="store", type=float, default=1.)
    parser.add_argument("--validation_pc", action="store", type=float, default=1.)
    parser.add_argument("--global_seed", action="store", type=int, default=12345)
    parser.add_argument("--seeds_path", action="store", type=str, default=None)
    parser.add_argument("--output_formats", action="store", nargs="*", default=["csv"], type=str)
    parser.add_argument("--num_shards", action="store", type=int, default=10)
    parser.add_argument("--idx_paths", type=ast.literal_eval, default=None)
    args = parser.parse_args()

    main(project_root=PROJECT_ROOT, **vars(args))

