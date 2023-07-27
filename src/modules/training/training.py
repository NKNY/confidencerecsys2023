### Tensorflow training procedure from config

import tensorflow as tf

import src.modules.models.models as models
import src.modules.utils.utils as utils
import src.modules.training.data_loading as data_loading

def restore_best_weights_if_callback(model, callbacks):
    for callback in callbacks:
        if isinstance(callback, tf.keras.callbacks.EarlyStopping) and callback.restore_best_weights:
            model.set_weights(callback.best_weights)
            print(f"Restored best weights from epoch {callback.best_epoch + 1}.")

def train(model, train_data, validation_data=None, training_params={}, **kwargs):
    # Preprocess training params
    _training_params = utils.preprocess_params(**training_params)
    results = model.fit(train_data, validation_data=validation_data, **_training_params, **kwargs)
    if "callbacks" in _training_params:
        restore_best_weights_if_callback(model, callbacks=_training_params["callbacks"])
    return results

def run_on_test_end(model, evaluation_params={}, prefix="test_", **kwargs):
    _evaluation_params = utils.preprocess_params(**evaluation_params)
    results = {}
    for callback in _evaluation_params["callbacks"]:
        if hasattr(callback, "run_on_test_end") and callback.run_on_test_end:
            callback.model = model
            callback.on_test_end(logs=results)
    return {prefix+k: v for k, v in results.items()}

def train_run(model_params, compilation_params, data_params, training_params, evaluation_params={}, meta_params = {}, **kwargs):
    # Input: dicts
    if "mixed_precision_policy" in meta_params:
        utils.set_mixed_precision_policy(name=meta_params["mixed_precision_policy"])
        print(tf.keras.mixed_precision.global_policy())

    if "run_log_path" in meta_params:
        utils.save_mixed_precision_policy(meta_params["run_log_path"])

    # Init model
    model = models.init_compile_model(model_params, compilation_params)

    # If using pretrained model, transfer its weights
    if "pretrained_model" in training_params:
        pretrained_model = training_params.pop("pretrained_model")
        print("PRETRAINED_MODEL", pretrained_model)
        if "dir" in pretrained_model and pretrained_model["dir"] is not None:
            pretrained_model_dir = utils.format_pretrained_model_dir(**pretrained_model)
            print("Starting weight transfer.")
            model = utils.transfer_model_weights(model, pretrained_model_dir)

    # Init training, validation and testing data
    data = data_loading.load_data(data_params)

    # Train model
    train(model, data["train"], data["validation"], training_params)  # results is tf History object

    # Run the post-training routine (e.g. export predictions)
    run_on_test_end(model, evaluation_params, prefix="test_")

    return model

class RankingEvaluationCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset=None, metrics=None, name="", params=None, run_on_test_end=False, *args, **kwargs):
        """
        Take data, group on uid, get 1 batch per user, combine multiple batches and flatten
        (variable batch size s.t. num uid's per batch is the same), predict.
        Then convert both pred and target to Ragged using mapping obtained during data loading.
        Finally, feed the Ragged tensors to the metrics, assumed to be formatted as metric_name: [obj1, obj2...]
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.dataset = dataset
        self.metrics = metrics
        self.run_on_test_end = run_on_test_end
        if params is not None:
            self.init_from_params(params)

    def init_from_params(self, d):
        if "metrics" in d:
            print("Preproc metrics")
            self.metrics = utils.preprocess_params(metrics=d["metrics"])["metrics"]
            print(self.metrics)
        if "data_path" in d:
            if isinstance(d["data_path"], str):
                print("Preproc ranking data")
                self.dataset = data_loading.load_ranking_data_csv(d["data_path"], d["batch_size"])
        if "run_on_test_end" in d:
            self.run_on_test_end = d["run_on_test_end"]
        if "name" in d:
            self.name = d["name"]

    def reset_metrics_states(self):
        for output, metrics in self.metrics.items():
            for metric in metrics:
                metric.reset_state()

    @staticmethod
    def re_rag(tensor, row_splits):
        return tf.RaggedTensor.from_row_splits(tensor, row_splits)

    @tf.function(reduce_retracing=True)
    def eval_step(self, X, y_true, row_splits):
        outputs = self.model.predict_step(X)
        y_true = self.re_rag(y_true, row_splits)
        for output, metrics in self.metrics.items():
            y_pred = self.re_rag(outputs[output], row_splits)
            for metric in metrics:
                metric(y_true, y_pred)

    def on_epoch_end(self, epoch, logs={}):
        self.reset_metrics_states()

        for i, (X, y_true, row_splits) in enumerate(self.dataset):
            self.eval_step(X, y_true, row_splits)

        logs.update(
            {f"{self.name}" + ("_" if len(self.name) else "") + metric.name: metric.result().numpy() for m, metrics in
             self.metrics.items() for metric in metrics})

    def on_test_end(self, logs=None):
        self.on_epoch_end(-1, logs)
