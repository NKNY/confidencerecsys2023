import tensorflow as tf

from tensorflow import keras
from keras import layers

class ConfidenceAwareMFRegularizer():
    def __init__(self, regularizer, user, item, general):
        if isinstance(regularizer, str):
            if regularizer == "l2":
                regularizer_class = tf.keras.regularizers.L2
        self.uid = regularizer_class(user)
        self.iid = regularizer_class(item)
        self.general = general

class ConfidenceAwareMFPrecisionRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, prior_a=1e-3, prior_b=1e-3, num_batches=1044.):
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.num_batches = num_batches

    def __call__(self, x, *args, **kwargs):
        return 2./self.num_batches*(self.prior_b*tf.reduce_sum(x) - (self.prior_a - 1)*tf.reduce_sum(tf.math.log(x)))


class GlobalBiasAdd(layers.Layer):
    def __init__(self, global_bias, **kwargs):
        super().__init__(**kwargs)
        self.global_bias = global_bias

    def call(self, inputs):
        if tf.executing_eagerly():
            print("EAGER")
        return inputs + self.global_bias

def init_model(num_users: int, num_items: int, num_hidden: int,
               initializer, regularizer=ConfidenceAwareMFRegularizer("l2", 1., 1., 0.01),
               bias_initializer="zeros", global_bias=3.,
               precision_regularizer=ConfidenceAwareMFPrecisionRegularizer(),
               logistic_mode=0,
               bin_size=1., min_rating=1., max_rating=5.,
               *args, **kwargs):

    # Get each input to the model
    uid_input = keras.Input(shape=(), name="uid")
    iid_input = keras.Input(shape=(), name="iid")

    # Embed each uid and iid
    uid_features = layers.Embedding(
        num_users, num_hidden, name="uid_features", embeddings_regularizer=regularizer.uid,
        embeddings_initializer=initializer["initializer"].from_config(initializer["config"])
    )(uid_input)
    iid_features = layers.Embedding(
        num_items, num_hidden, name="iid_features", embeddings_regularizer=regularizer.iid,
        embeddings_initializer=initializer["initializer"].from_config(initializer["config"])
    )(iid_input)

    uid_bias = layers.Embedding(num_users, 1, name="uid_bias",
                                embeddings_initializer=bias_initializer)(uid_input)
    iid_bias = layers.Embedding(num_items, 1, name="iid_bias",
                                embeddings_initializer=bias_initializer)(iid_input)

    uid_std = layers.Embedding(num_users, 1, name="uid_std",
                                embeddings_initializer=tf.keras.initializers.Ones(),
                               embeddings_regularizer=precision_regularizer)(uid_input)
    iid_std = layers.Embedding(num_items, 1, name="iid_std",
                                embeddings_initializer=tf.keras.initializers.Ones(),
                               embeddings_regularizer=precision_regularizer)(iid_input)

    # Predictions (sans global bias, because wandb doesn't allow to just "add" a variable)
    pred_minus_global_bias = layers.Add(name="pred_minus_global")([layers.Dot(axes=(1, 1))(
        [uid_features, iid_features]), uid_bias, iid_bias]
    )
    # Global bias constraint
    pred = GlobalBiasAdd(global_bias, name="pred_")(pred_minus_global_bias)

    # Rescale
    if logistic_mode == 2:
        pred = layers.Lambda(lambda x: min_rating + (max_rating - min_rating)*tf.sigmoid(x), name="pred__")(pred)

    sample_weight = layers.Lambda(lambda x: x[0]*x[1]*x[2], name="sample_weight")([uid_std, iid_std, regularizer.general])
    y_pred = layers.Lambda(lambda x: tf.concat(x, axis=-1), name="y_pred")([pred, sample_weight])
    pred = layers.Lambda(lambda x: x[:,0], name="mean")(pred)

    outputs = {"y_pred": y_pred, "pred": pred, "uid_std": uid_std, "iid_std": iid_std}

    # Create Model
    mf_model = keras.Model(
        inputs=[uid_input, iid_input],
        outputs=outputs
    )

    return mf_model
