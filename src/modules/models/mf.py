import tensorflow as tf

from tensorflow import keras
from keras import layers

class GlobalBiasAdd(layers.Layer):
    def __init__(self, global_bias, **kwargs):
        super().__init__(**kwargs)
        self.global_bias = tf.Variable(global_bias, trainable=True)
        print("GLOBAL BIAS", self.global_bias.dtype)

    def call(self, inputs):
        if tf.executing_eagerly():
            print("EAGER")
        return inputs + self.global_bias  # This breaks if dtype is wrong e.g. because wandb made float an int.

def init_model(num_users: int, num_items: int, num_hidden: int,
               regularizer, initializer, bias_initializer="zeros", global_bias=3.5, regularize_activity=True,
               logistic_mode=0, bin_size=1., min_rating=1., max_rating=5.,
               var=1.,
               *args, **kwargs):

    # Get each input to the model
    uid_input = keras.Input(shape=(), name="uid")
    iid_input = keras.Input(shape=(), name="iid")

    # Embed each uid and iid
    uid_features = layers.Embedding(
        num_users, num_hidden, name="uid_features",
        embeddings_regularizer=(regularizer if not regularize_activity else None),
        activity_regularizer=(regularizer if regularize_activity else None),
        embeddings_initializer=initializer["initializer"].from_config(initializer["config"])
    )(uid_input)
    iid_features = layers.Embedding(
        num_items, num_hidden, name="iid_features",
        embeddings_regularizer=(regularizer if not regularize_activity else None),
        activity_regularizer=(regularizer if regularize_activity else None),
        embeddings_initializer=initializer["initializer"].from_config(initializer["config"])
    )(iid_input)

    uid_bias = layers.Embedding(num_users, 1, name="uid_bias",
                                embeddings_initializer=bias_initializer)(uid_input)
    iid_bias = layers.Embedding(num_items, 1, name="iid_bias",
                                embeddings_initializer=bias_initializer)(iid_input)

    # Predictions (sans global bias, because wandb doesn't allow to just "add" a variable)
    pred_minus_global_bias = layers.Add(name="pred_minus_global")([layers.Dot(axes=(1, 1))(
        [uid_features, iid_features]), uid_bias, iid_bias]
    )

    if logistic_mode == 2:
        pred = GlobalBiasAdd(global_bias, name="pred__")(pred_minus_global_bias)
        pred = layers.Lambda(lambda x: min_rating + (max_rating - min_rating)*tf.sigmoid(x), name="pred")(pred)
        pred_ = layers.Lambda(lambda x: x[:, 0], name="pred_")(pred)
    else:
        # Global bias constraint
        pred = GlobalBiasAdd(global_bias, name="pred")(pred_minus_global_bias)
        pred_ = layers.Lambda(lambda x: x[:, 0], name="pred_")(pred)

    bins_mass = NormalBinsMass(bin_size, min_rating, max_rating, name="normal_bins_mass")([pred, var])

    outputs = {"pred": pred, "pred_": pred_, "bins_mass": bins_mass}

    # Create MF
    mf_model = keras.Model(
        inputs=[uid_input, iid_input],
        outputs=outputs
    )

    return mf_model

class NormalBinsMass(layers.Layer):
    def __init__(self, bin_size=1., min_rating=1, max_rating=5, **kwargs):
        # num_bins should be something divisible by 5 s.t. the model can achieve 0 loss
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.num_bins = (self.max_rating - self.min_rating) / self.bin_size + 1
        self.bins = tf.range(self.min_rating, self.max_rating + bin_size, bin_size)  # 0.5, 1, 1.5 ... 4.5, 5 (10)
        self.bins_ends = self.bins[:-1] + self.bin_size / 2.  # 0.75, 1.25, ... 4.25, 4.75 (9)

    @staticmethod
    def cdf(x, mean, var):
        return 0.5*(1 + tf.math.erf((x - mean)/(tf.sqrt(2*var))))

    def call(self, inputs):
        mean, var = inputs
        # Calculate cdf at each bin end: (None, num_bins)
        cdf = self.cdf(self.bins_ends, mean, var)
        # Replace first and last cdf bin to avoid issues with calculating d/dx of cdf with convex tails
        cdf = tf.concat([tf.zeros_like(mean), cdf, tf.ones_like(mean)], axis=-1)
        # Calculate mass in each bin: (None, num_bins)
        mass = tf.experimental.numpy.diff(cdf)
        # Output tensors: prediction, mass
        return mass