import json
import os
import pickle
import sys

import numpy as np
import tensorflow as tf

from tensorflow import keras

import src.modules.losses.distribution_losses as distribution_losses
import src.modules.losses.pointwise_losses as pointwise_losses
import src.modules.metrics.ranking.metrics as ranking_metrics
import src.modules.metrics.rating_metrics as rating_metrics
import src.modules.models.models as models
import src.modules.models.baselines.CMF as CMF
import src.modules.models.LBD as LBD
import src.modules.training.training as training
import src.modules.training.data_loading as data_loading

def set_mixed_precision_policy(name):
    policy = tf.keras.mixed_precision.Policy(name)
    tf.keras.mixed_precision.set_global_policy(policy)

def init_compilation_params(conversion_table: dict, d: dict):
    # No longer overwrites the original dict d
    output = {}
    for k, v in d.items():
        if v in conversion_table:
            output[k] = conversion_table[v]()
        elif not v.islower():
            output[k] = eval(v)()
        else:
            output[k] = v
    return output

def custom_eval(obj, **kwargs):
    return eval(obj)(**kwargs)

def is_tuple(x):
    return isinstance(x, tuple)
def is_list(x):
    return isinstance(x, list)

def preprocess_regularizer(regularizer):
    for prop in ["uid", "iid", "general"]:
        if not hasattr(regularizer, prop):
            setattr(regularizer, prop, regularizer)
    return regularizer

def preprocess_params(**params):

    # All used metrics and losses should be imported here
    conversion_table = {
        "regularizer": ("regularizer", lambda x: preprocess_regularizer(custom_eval(**x))),
        "initializer": ("initializer", lambda x: {"initializer": eval(x["obj"]), "config": x["config"]}),
        "optimizer": ("optimizer", lambda x: custom_eval(**x)),
        "upsilon_layer_id": ("upsilon_layer", lambda x: LBD.get_upsilon_layer(x)),
        "bias_initializer": ("bias_initializer", lambda x: custom_eval(**x)),
        "bias_initializers": ("bias_initializers", lambda x: [(conversion_table[o], c) for o, c in x]),

        # Training params
        "callbacks": ("callbacks", lambda x: [conversion_table[k](v) if k in conversion_table else v for d in x for k, v in d.items()]),

        # Losses and metrics are looked up on the basis of the value, not the key. The key stays unchanged.
        # Losses and metrics assumed to be passed inside a list to allow multiple for single output.
        # Losses and metrics entries in a list either individual strings or len-2 lists with string and
        # dict with params to be passed to object denoted by the string
        "loss": ("loss", lambda tensor_losses: {
            # Tensor: [loss1_name, loss2_name, (loss3_name, {k1_passed_to_loss3: v1}) ... ]
            tensor: [
                conversion_table[loss[0]](**loss[1]) if is_tuple(loss) or is_list(loss)  # [tensor1: [("AdaptedXent", {"bin_size": 1}), ...]
                else conversion_table[loss]() if loss in conversion_table  # [tensor1: ["rmse"]...]
                else loss  # [tensor1: [keras.metric.Metric()]]
                for loss in losses] for tensor, losses in tensor_losses.items()
        }),
        "metrics": ("metrics", lambda tensor_metrics: {
            # Tensor: [metric1_name, metric2_name, (metric3_name, {k1_passed_to_metric3: v1}) ... ]
            tensor: [
                conversion_table[metric[0]](**metric[1]) if is_tuple(metric) or is_list(metric)  # [tensor1: [("AdaptedAcc", {"bin_size": 1}), ...]
                else conversion_table[metric]() if metric in conversion_table  # [tensor1: ["rmse"]...]
                else metric  # [tensor1: [keras.metric.Metric()]]
                for metric in metrics] for tensor, metrics in tensor_metrics.items()
        }),

        # Losses
        "distribution_losses.DiscretizedMSE": distribution_losses.DiscretizedMSE,
        "keras.losses.MeanSquaredError": pointwise_losses.CustomRMSE,
        "mse": pointwise_losses.CustomRMSE,
        "distribution_losses.AdaptedCrossentropy": distribution_losses.AdaptedCrossentropy,
        "pointwise_losses.WeightedMSEWithWeightPenalty": pointwise_losses.WeightedMSEWithWeightPenalty,

        # Metrics
        "rmse": rating_metrics.CustomRMSEMetric,
        "mae": keras.metrics.MeanAbsoluteError,
        "rating_metrics.AdaptedCrossentropy": rating_metrics.AdaptedCrossentropy,
        "rating_metrics.AdaptedAccuracy": rating_metrics.AdaptedAccuracy,
        "ranking_metrics.NDCGMetric": ranking_metrics.NDCGMetric,
        "ranking_metrics.DCGMetric": ranking_metrics.DCGMetric,

        # Training
        "EarlyStopping": lambda x: tf.keras.callbacks.EarlyStopping(**x),
        "RankingEvaluationCallback": lambda x: training.RankingEvaluationCallback(**x),
        "ExportEvalPredictionsCallback": lambda x: ExportEvalPredictionsCallback(**x),
        "ModelCheckpoint": lambda x: tf.keras.callbacks.ModelCheckpoint(**x),
        "TerminateOnNaN": lambda x: tf.keras.callbacks.TerminateOnNaN(**x),

        # Initializers
        "keras.initializers.Ones": keras.initializers.Ones,
        "keras.initializers.Constant": keras.initializers.Constant,
        "keras.initializers.Zeros": keras.initializers.Zeros,
        "keras.initializers.RandomNormal": keras.initializers.RandomNormal,

        # Custom fields for regularizers
        "precision_regularizer": ("precision_regularizer", lambda x: preprocess_regularizer(custom_eval(**x))),

        # Custom list of initializers formatted as {"obj": object, "config": {"a": "b", "c": "d"}}
        "initializers": ("initializers", lambda x: {k: conversion_table["initializer"][1](v) for k, v in x.items()}),

        # Custom list of regularizers formatted as {"obj": object, "config": {"a": "b", "c": "d"}}
        "regularizers": ("regularizers", lambda x: {k:conversion_table["regularizer"][1](v) for k, v in x.items()}),
    }

    _params = {}
    k_v = list(params.items())
    for k, v in k_v:
        if k == "loss":
            print(k, v)
        do_convert = k in conversion_table
        if do_convert:
            new_k, conversion_fn = conversion_table[k]
            _params[new_k] = conversion_fn(v)
        else:
            _params[k] = v

    print("PARAMS: ")
    print(_params)
    return _params



def nested_dict_values(d):
  for v in d.values():
    if isinstance(v, dict):
      yield from nested_dict_values(v)
    else:
      yield v


def load_pretrained_model(pretrained_model_config_path, pretrained_model_weight_path, pretrained_model_dtype=None):
    # Further assumes that the `pretrained_model_weight_path`'s parent directory contains `variables.data*` files,
    # and `variables.index`.
    if pretrained_model_dtype is not None:
        old_policy = tf.keras.mixed_precision.global_policy()
        set_mixed_precision_policy(name=pretrained_model_dtype)
        print(f"Temporarily set mixed precision policy from {old_policy.variable_dtype} to "
              f"{tf.keras.mixed_precision.global_policy().variable_dtype}")

    pretrained_model_params = json.load(open(pretrained_model_config_path))
    print(f"Init/compile pretrained model from {pretrained_model_config_path}.")
    model = models.init_compile_model(
        pretrained_model_params["model_params"],
        pretrained_model_params["compilation_params"]
    )
    model.load_weights(pretrained_model_weight_path)

    if pretrained_model_dtype is not None:
        set_mixed_precision_policy(name=old_policy.variable_dtype)
        print(f"Returned mixed precision policy to {tf.keras.mixed_precision.global_policy().variable_dtype}")

    return model

def get_model_mixed_precision_policy_dtype(pretrained_model_dir):
    policies = ["float16", "float32", "float64"]
    policy_paths = {p: os.path.join(pretrained_model_dir, p) for p in policies}
    for p, path in policy_paths.items():
        if os.path.isfile(path):
            return p
    return None

def save_mixed_precision_policy(dir):
    policies = ["float16", "float32", "float64"]
    for p in policies:
        if p in str(tf.keras.mixed_precision.global_policy()):
            with open(os.path.join(dir, p), "w") as output_file:
                output_file.write("")

def transfer_model_weights(model, pretrained_model_dir):
    # Assumes both models have the same names
    # Further assumes that the `pretrained_model_dir` contains `variables/variables.data*` files,
    # `variables/variables.index`, as well as a `config.json` sufficient to instantiate the pretrained model before
    # transferring its weights to `model`.
    # Furthermore, if current global mixed precision policy is not the same dtype as the dtype of the saved model,
    # `pretrained_model_dir` has to contain a file with name `float16/32/64` indicating which dtype it is in.
    pretrained_model_config_path = os.path.join(pretrained_model_dir, "config.json")
    pretrained_model_weight_path = os.path.join(pretrained_model_dir, "variables", "variables")
    pretrained_model_dtype = get_model_mixed_precision_policy_dtype(pretrained_model_dir)
    pretrained_model = load_pretrained_model(pretrained_model_config_path, pretrained_model_weight_path,
                                             pretrained_model_dtype)
    for i, x in enumerate(pretrained_model.weights):
        for j, x2 in enumerate(model.weights):
            if x.name == x2.name:
                print(f"Loading pretrained {x.name}")
                model.weights[j].assign(tf.cast(x, x2.dtype))
    return model



class ExportEvalPredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self,
                 validation_data=None,
                 export_path=None,
                 include_X=True,
                 include_y=True,
                 params=None,
                 batch_size=1,
                 run_on_test_end=False,
                 *args, **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.validation_data = validation_data
        self.export_path = export_path
        self.include_X = include_X
        self.include_y = include_y
        self.batch_size = batch_size
        if params is not None:
            self.init_from_params(params)

    def init_from_params(self, d):
        if "export_path" in d:
            print("Setting export path")
            self.export_path = os.path.join(d["export_path"], "export")
        if "data_path" in d:
            if isinstance(d["data_path"], str):
                print("Preproc ExportEvalPredictionsCallback data")
                self.validation_data = data_loading.load_ranking_data_csv(d["data_path"], d["batch_size"])
        if "include_X" in d:
            self.include_X = d["include_X"]
        if "include_y" in d:
            self.include_y = d["include_y"]
        if "batch_size" in d:
            self.batch_size = d["batch_size"]
        if "run_on_test_end" in d:
            self.run_on_test_end = d["run_on_test_end"]

    def on_test_end(self, logs=None):
        p = self.model.predict(self.validation_data, batch_size=self.batch_size)
        if self.include_X:
            x_keys = self.validation_data.element_spec[0].keys()
            p["X"] = {x_key: np.concatenate([x[0][x_key].numpy() for x in self.validation_data]) for x_key in x_keys}
        if self.include_y:
            p["rating"] = np.concatenate([x[1].numpy() for x in self.validation_data])
        with open(self.export_path, "wb") as output_file:
            print(f"Dumping results to {self.export_path}.")
            pickle.dump(p, output_file)

def format_pretrained_model_dir(dir, split_outer, split_inner, *args, **kwargs):
    print("HERE", os.path.join(dir, f"{os.path.split(dir)[-1]}-{split_outer}-{split_inner}"))
    return os.path.join(dir, f"{os.path.split(dir)[-1]}-{split_outer}-{split_inner}")