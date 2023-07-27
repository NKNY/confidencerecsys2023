import math

import tensorflow as tf

from tensorflow import keras
from keras import layers

class GlobalBiasAdd(layers.Layer):
    def __init__(self, global_bias, **kwargs):
        super().__init__(**kwargs)
        self.global_bias = global_bias

    def call(self, inputs):
        if tf.executing_eagerly():
            print("EAGER")
        return inputs + self.global_bias

def init_initializers(thresholds_use_items, t1_mode, b_mode, use_biases, bin_size, min_rating, max_rating=5.,override : dict = {}):
    initializers = {}
    for k in ["uid_t1", "iid_t1", "uid_beta", "iid_beta", "uid_bias", "iid_bias"]:
        initializers[k] = {"initializer": keras.initializers.Constant, "config": {"value": None}}

    if not thresholds_use_items:
        # Init t1 to 0.75
        initializers["uid_t1"]["config"]["value"] = min_rating + 0.5 * bin_size
        initializers["iid_t1"]["config"]["value"] = min_rating + 0.5 * bin_size
        # Init betas to log(bin_size)
        initializers["uid_beta"]["config"]["value"] = math.log(bin_size)
        initializers["iid_beta"]["config"]["value"] = math.log(bin_size)

        # Setting each item bias to the average ML-10m rating
        initializers["uid_bias"]["config"]["value"] = 0  # This will be ignored if `use_biases=False`
        initializers["iid_bias"]["config"]["value"] = 3.5

    else:
        # If t1_mode = 1:
        if t1_mode == 1:
            # uid_t1 = iid_t1 = 0.375
            initializers["uid_t1"]["config"]["value"] = 0.5 * (min_rating + 0.5 * bin_size)
            initializers["iid_t1"]["config"]["value"] = 0.5 * (min_rating + 0.5 * bin_size)
        # If t1_mode = 2:
        elif t1_mode == 2:
            # uid_t1 = iid_t1 = sqrt(0.75)
            initializers["uid_t1"]["config"]["value"] = math.sqrt(min_rating + 0.5 * bin_size)
            initializers["iid_t1"]["config"]["value"] = math.sqrt(min_rating + 0.5 * bin_size)
        # If b_mode = 1:
        if b_mode == 1:
            # b = log(bin_size/2)
            initializers["uid_beta"]["config"]["value"] = math.log(0.5 * bin_size)
            initializers["iid_beta"]["config"]["value"] = math.log(0.5 * bin_size)
        # If b_mode = 2:
        elif b_mode == 2:
            # b = 0.5 log(bin_size)
            initializers["uid_beta"]["config"]["value"] = 0.5 * math.log(bin_size)
            initializers["iid_beta"]["config"]["value"] = 0.5 * math.log(bin_size)

        # Setting each user and item bias to half of average ML-10m rating
        initializers["uid_bias"]["config"]["value"] = 1.75
        initializers["iid_bias"]["config"]["value"] = 1.75

    # Default embeddings
    initializers["uid_features"] = {"initializer": keras.initializers.RandomNormal, "config": {"stddev": 0.01}}
    initializers["iid_features"] = {"initializer": keras.initializers.RandomNormal, "config": {"stddev": 0.01}}

    # Add all
    for k, v in override.items():
        initializers[k] = v

    for k, v in initializers.items():
        initializers[k] = v["initializer"].from_config(v["config"])

    return initializers

def init_regularizers(override={}):

    # Defaults
    regularizers = {
        "uid_features": keras.regularizers.L2(0.04),
        "iid_features": keras.regularizers.L2(0.04),
        "uid_bias": keras.regularizers.L2(0.),
        "iid_bias": keras.regularizers.L2(0.),
        "uid_beta": keras.regularizers.L2(0.001),
        "iid_beta": keras.regularizers.L2(0.001)}

    regularizers["uid_t1"] = regularizers["uid_bias"]
    regularizers["iid_t1"] = regularizers["iid_bias"]

    # Add all
    for k, v in override.items():
        regularizers[k] = v

    return regularizers


def init_model(num_users: int, num_items: int, num_hidden: int,
               initializers: dict = {},
               regularizers: dict = {},
               thresholds_use_item: bool = False,
               t1_mode: int = 1,
               beta_mode: int = 1,
               use_biases: bool = True,
               bin_size=0.5, min_rating=0.5, max_rating=5.,
               *args, **kwargs):

    num_bins = int((max_rating - min_rating) / bin_size) + 1
    bins_params = {"bin_size": bin_size, "min_rating": min_rating, "max_rating": max_rating}

    initializers = init_initializers(thresholds_use_item, t1_mode, beta_mode, use_biases, **bins_params, override=initializers)
    regularizers = init_regularizers(override=regularizers)

    # Get each input to the model
    uid_input = keras.Input(shape=(), name="uid")
    iid_input = keras.Input(shape=(), name="iid")

    # Features
    uid_features = layers.Embedding(num_users, num_hidden, name="uid_features",
                                    activity_regularizer=regularizers["uid_features"],
                                    embeddings_initializer=initializers["uid_features"])(uid_input)
    iid_features = layers.Embedding(num_items, num_hidden, name="iid_features",
                                    activity_regularizer=regularizers["iid_features"],
                                    embeddings_initializer=initializers["iid_features"])(iid_input)

    # Biases
    uid_bias = layers.Embedding(num_users, 1, name="uid_bias",
                                    activity_regularizer=regularizers["uid_bias"],
                                    embeddings_initializer=initializers["uid_bias"])(uid_input)
    iid_bias = layers.Embedding(num_items, 1, name="iid_bias",
                                    activity_regularizer=regularizers["iid_bias"],
                                    embeddings_initializer=initializers["iid_bias"])(iid_input)

    # T1's
    uid_t1 = layers.Embedding(num_users, 1, name="uid_t1",
                                    activity_regularizer=regularizers["uid_t1"],
                                    embeddings_initializer=initializers["uid_t1"])(uid_input)
    iid_t1 = layers.Embedding(num_items, 1, name="iid_t1",
                                    activity_regularizer=regularizers["iid_t1"],
                                    embeddings_initializer=initializers["iid_t1"])(iid_input)

    # Betas
    uid_beta = layers.Embedding(num_users, num_bins-2, name="uid_beta",
                                    activity_regularizer=regularizers["uid_beta"],
                                    embeddings_initializer=initializers["uid_beta"])(uid_input)
    iid_beta = layers.Embedding(num_items, num_bins-2, name="iid_beta",
                                    activity_regularizer=regularizers["iid_beta"],
                                    embeddings_initializer=initializers["iid_beta"])(iid_input)

    uTi = layers.Dot(axes=(1, 1))([uid_features, iid_features])

    # Determine model score
    if thresholds_use_item and (not use_biases):
        y_ui = uTi
    elif (not thresholds_use_item) and (not use_biases):
        y_ui = layers.Add(name="dot_plus_b_i")([uTi, iid_bias])
    else:
        y_ui = layers.Add(name="dot_plus_b_u_b_i")([uTi, iid_bias, uid_bias])

    # Determine t_1
    if not thresholds_use_item:
        t1 = uid_t1
    elif thresholds_use_item and t1_mode == 1:
        t1 = layers.Add(name="t1_sum")([uid_t1, iid_t1])
    elif thresholds_use_item and t1_mode == 2:
        t1 = layers.Multiply(name="t1_prod")([uid_t1, iid_t1])
    # Determine gaps
    if not thresholds_use_item:
        beta = layers.Lambda(lambda x: tf.exp(x), name="beta")(uid_beta)
    elif thresholds_use_item and beta_mode == 1:
        beta = layers.Lambda(lambda x: tf.exp(x[0]) + tf.exp(x[1]), name="beta_sum")([uid_beta, iid_beta])
    elif thresholds_use_item and beta_mode == 2:
        beta = layers.Lambda(lambda x: tf.exp(x[0] + x[1]), name="beta_sum")([uid_beta, iid_beta])

    # Determine t_2 ... t_{N-1} and concat with t1
    beta_ext = layers.Concatenate(axis=-1, name='beta_ext')([tf.zeros_like(t1), beta])
    beta_cum = layers.Lambda(lambda x: tf.cumsum(x, axis=-1), name="beta_cum")(beta_ext)
    T = layers.Add(name="T")([t1, beta_cum])

    # Get CDF for t_1 ... t_{N-1}
    sigmoid_inputs = layers.Subtract(name="sigmoid_inputs")([T, y_ui])
    sigmoid = layers.Activation(activation="sigmoid", name="sigmoid")(sigmoid_inputs)

    # Add t_0 and t_N
    cdf = layers.Concatenate(axis=-1, name="cdf")([tf.zeros_like(t1), sigmoid, tf.ones_like(t1)])

    # Get bin scores
    bins_mass = layers.Lambda(lambda x: x[:,1:] - x[:,:-1], name="bins_mass")(cdf)

    # Get point predictors
    bins_mean = BinsMean(**bins_params, name="bins_mean")(bins_mass)
    bins_mode = BinsMode(**bins_params, name="bins_mode")(bins_mass)

    # Gather outputs, including the binned distribution
    outputs = {"bins_mass": bins_mass, "bins_mean": bins_mean, "bins_mode": bins_mode}

    ordrec_model = keras.Model(
        inputs=[uid_input, iid_input],
        outputs=outputs
    )

    return ordrec_model

class BinsMode(layers.Layer):
    def __init__(self,  bin_size=1., min_rating=1., max_rating=5., **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def call(self, inputs):
        output = (self.min_rating + tf.cast(tf.argmax(inputs, axis=-1), dtype=tf.float32)*self.bin_size)
        return output

class BinsMean(layers.Layer):
    def __init__(self,  bin_size=1., min_rating=1., max_rating=5., **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def call(self, inputs):
        output = tf.reduce_sum(tf.multiply(inputs, tf.range(self.min_rating, self.max_rating+self.bin_size, self.bin_size, dtype=inputs.dtype)), axis=-1)
        return output