from typing import Literal

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from keras import layers

def get_upsilon_layer(id: int):
    upsilon_layers = {
        1: layers.Lambda(
            lambda x: tf.maximum(layers.Multiply()([tf.norm(x[0], axis=1, keepdims=True), tf.norm(x[1], axis=1, keepdims=True)]), 1e-6),
            name="upsilon1"),  # Norm
        2: layers.Lambda(lambda x: tf.maximum(tf.abs(tf.reduce_sum(tf.multiply(x[0],x[1]), axis=-1, keepdims=True)), 1e-6), name="upsilon2"),  # Dot
        3: layers.Lambda(lambda x: tf.maximum(tf.norm(x[0] + x[1], axis=1, keepdims=True), 1e-6), name="upsilon3"),  # Sum
    }
    return upsilon_layers[id]


def init_model(
        num_users: int, num_items: int,
        num_hidden: int, upsilon_layer,
        initializer, regularizer,
        bin_size=1., min_rating=1., max_rating=5.,
        bias_mode=0, bias_initializers=[(tf.keras.initializers.Ones, {})]*2,
        regularize_activity=True,
        split_embeddings=False, adaptive_edges=False,
        *args, **kwargs
):

    # Get each input to the model
    uid_input, iid_input = keras.Input(shape=(), name="uid"), keras.Input(shape=(), name="iid")

    # Embed each uid and iid
    uid_features = layers.Embedding(num_users, num_hidden+1, name="uid_features",
                                    embeddings_regularizer=(regularizer if not regularize_activity else None),
                                    activity_regularizer=(regularizer if regularize_activity else None),
                                    embeddings_initializer=initializer["initializer"].from_config(initializer["config"])
                                    )(uid_input)
    iid_features = layers.Embedding(num_items, num_hidden+1, name="iid_features",
                                    embeddings_regularizer=(regularizer if not regularize_activity else None),
                                    activity_regularizer=(regularizer if regularize_activity else None),
                                    embeddings_initializer=initializer["initializer"].from_config(initializer["config"])
                                    )(iid_input)

    # Embed each uid and iid
    uid_confidence_features = layers.Embedding(num_users, num_hidden + 1, name="uid_confidence_features",
                                               embeddings_regularizer=(regularizer if not regularize_activity else None),
                                               activity_regularizer=(regularizer if regularize_activity else None),
                                               embeddings_initializer=initializer["initializer"].from_config(
                                                   initializer["config"])
                                               )(uid_input) if split_embeddings else uid_features
    iid_confidence_features = layers.Embedding(num_items, num_hidden + 1, name="iid_confidence_features",
                                               embeddings_regularizer=(regularizer if not regularize_activity else None),
                                               activity_regularizer=(regularizer if regularize_activity else None),
                                               embeddings_initializer=initializer["initializer"].from_config(
                                        initializer["config"])
                                    )(iid_input) if split_embeddings else iid_features

    # Forward steps
    dot = layers.Dot(axes=(1, 1), name="dot")([uid_features[:,:-1], iid_features[:,:-1]])  # u·i
    norm_layer = layers.Lambda(lambda x: tf.norm(x, axis=1, keepdims=True), name="norm_layer")
    uid_norm, iid_norm = norm_layer(uid_features[:,:-1]), norm_layer(iid_features[:,:-1])  # ||u||, ||i||
    len_prod = layers.Multiply(name="len_prod")([uid_norm, iid_norm])  # ||u||·||i||

    mu = layers.Lambda(lambda x: tf.clip_by_value(0.5 + 0.5 * x[0] / tf.maximum(x[1], 1e-6), 1e-6, 1-1e-6), name="mu")([dot, len_prod])
    upsilon = upsilon_layer([uid_confidence_features[:, :-1], iid_confidence_features[:, :-1]])

    if bias_mode == 2:
        mu_initializer = [tf.initializers.RandomUniform, None]
        upsilon_initializer = [tf.initializers.RandomUniform, None]
        mu_initializer[1] = {"minval": tf.sqrt(0.5)-0.1, "maxval": tf.sqrt(0.5)+0.1}
        upsilon_initializer[1] = {"minval": 1-0.1, "maxval": 1+0.1}
        uid_mu_emb = layers.Embedding(num_users, 1, mu_initializer[0].from_config(mu_initializer[1]), name="uid_mu_bias")(uid_input)
        uid_upsilon_emb = layers.Embedding(num_users, 1, upsilon_initializer[0].from_config(upsilon_initializer[1]), name="uid_upsilon_bias")(uid_input)
        iid_mu_emb = layers.Embedding(num_items, 1, mu_initializer[0].from_config(mu_initializer[1]), name="iid_mu_bias")(iid_input)
        iid_upsilon_emb = layers.Embedding(num_items, 1, upsilon_initializer[0].from_config(upsilon_initializer[1]), name="iid_upsilon_bias")(iid_input)
        upsilon = layers.Lambda(lambda x: tf.clip_by_value(x[0]*x[1]*x[2], 1e-6, 15.))([upsilon, uid_upsilon_emb, iid_upsilon_emb])
        l = lambda x: tf.clip_by_value(
            tf.cast(x[0] < x[1],tf.float32)*(0.5 * x[0] / tf.maximum(x[1], 1e-6)) +
            (1.-tf.cast(x[0]<x[1], tf.float32))*(0.5 + 0.5 * (x[0] - x[1]) / tf.maximum(1 - x[1], 1e-6)),
            1e-6, 1-1e-6)
        mu = layers.Lambda(l)([mu, uid_mu_emb * iid_mu_emb])

    alpha = layers.Lambda(lambda x: tf.maximum(x[0]*x[1], 1e-2), name="alpha")([mu, upsilon])
    beta = layers.Lambda(lambda x: tf.maximum(x[0]-x[1], 1e-2), name="beta")([upsilon, alpha])  # Equivalent to (1-mu)*upsilon

    # Uid/iid learned alpha/beta adjustments
    if bias_mode == 1:
        initializers, configs = [x[0] for x in bias_initializers], [x[1] for x in bias_initializers]
        uid_alpha_emb = layers.Embedding(num_users, 1, initializers[0].from_config(configs[0]),name="uid_alpha_bias")(uid_input)
        uid_beta_emb = layers.Embedding(num_users, 1, initializers[1].from_config(configs[1]), name="uid_beta_bias")(uid_input)
        iid_alpha_emb = layers.Embedding(num_items, 1, initializers[0].from_config(configs[0]), name="iid_alpha_bias")(iid_input)
        iid_beta_emb = layers.Embedding(num_items, 1, initializers[1].from_config(configs[1]),name="iid_beta_bias")(iid_input)
        alpha = GlobalBiasAdd(0.3)(alpha)
        beta = GlobalBiasAdd(0.3)(beta)
        alpha = layers.Lambda(lambda x: tf.maximum(1e-2, x[0]+x[1]+x[2]))([alpha, uid_alpha_emb, iid_alpha_emb])
        beta = layers.Lambda(lambda x: tf.maximum(1e-2, x[0]+x[1]+x[2]))([beta, uid_beta_emb, iid_beta_emb])

    outputs = {"alpha": alpha, "beta": beta, "mu": mu, "upsilon": upsilon}

    if adaptive_edges:
        num_bins = tf.cast((max_rating - min_rating) / bin_size + 1, tf.int32)
        bins_size_terms_initializer = keras.initializers.ones()
        uid_bin_size_terms = keras.layers.Embedding(num_users, num_bins, name="uid_bin_size_terms",
                                                    embeddings_initializer=bins_size_terms_initializer)(uid_input)
        iid_bin_size_terms = keras.layers.Embedding(num_items, num_bins, name="iid_bin_size_terms",
                                                    embeddings_initializer=bins_size_terms_initializer)(iid_input)
        ui_bin_size_terms = keras.layers.Lambda(lambda x: tf.exp(x[0]+x[1]))([uid_bin_size_terms, iid_bin_size_terms])
        ui_bin_size_terms_norm = keras.layers.Lambda(lambda x: x / tf.reduce_sum(x, axis=-1, keepdims=True))(ui_bin_size_terms)
        edges = keras.layers.Lambda(lambda x: tf.cumsum(x, axis=-1), name="edges")(ui_bin_size_terms_norm)
        outputs["edges"] = edges

    metric_params = {"bin_size": bin_size, "min_rating": min_rating, "max_rating": max_rating}

    # Beta Outputs
    if not adaptive_edges:
        beta_bins_mass = BetaBinsMass(**metric_params, name="beta_bins_mass")([alpha, beta])
        beta_mean = BetaMean(**metric_params, name="beta_mean")([alpha, beta])
        beta_median = BetaMedian(**metric_params, name="beta_median")([alpha, beta])
        beta_mode = BetaMode(**metric_params, name="beta_mode")([alpha, beta])
        beta_bins_mode = BetaBinsMode(**metric_params, name="beta_bins_mode")(beta_bins_mass)
        beta_bins_mean = BetaBinsMean(**metric_params, name="beta_bins_mean")(beta_bins_mass)
        outputs.update({"bins_mass": beta_bins_mass, "mean": beta_mean, "mode": beta_mode, "median": beta_median,
                        "bins_mode": beta_bins_mode, "bins_mean": beta_bins_mean})
    else:
        beta_bins_mass = BetaBinsMassAdaptive(**metric_params, name="beta_bins_mass")([alpha, beta, edges])
        beta_bins_mean = BetaBinsMean(**metric_params, name="beta_bins_mean")(beta_bins_mass)
        beta_bins_mode = BetaBinsMode(**metric_params, name="beta_bins_mode")(beta_bins_mass)
        outputs.update({"bins_mass": beta_bins_mass, "bins_mean": beta_bins_mean, "bins_mode": beta_bins_mode})
    model = keras.Model(
        inputs=[uid_input, iid_input],
        outputs=outputs
    )
    return model

class GlobalBiasAdd(layers.Layer):
    def __init__(self, bias=1., **kwargs):
        super().__init__(**kwargs)
        self.global_bias = tf.Variable(bias, trainable=True)

    def call(self, inputs):
        return inputs + self.global_bias

class BetaBinsMode(layers.Layer):
    def __init__(self,  bin_size=1., min_rating=1., max_rating=5., **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def call(self, inputs):
        output = self.min_rating + tf.cast(tf.argmax(inputs, axis=-1), dtype=tf.float32)*self.bin_size
        return output

class BetaBinsMean(layers.Layer):
    def __init__(self,  bin_size=1., min_rating=1., max_rating=5., **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def call(self, inputs):
        output = tf.reduce_sum(tf.multiply(inputs, tf.range(self.min_rating, self.max_rating+self.bin_size, self.bin_size, dtype=inputs.dtype)), axis=-1)  # Will not work properly with bin_size!=1.
        return output

class BetaBinsMass(layers.Layer):
    def __init__(self, bin_size=1., min_rating=1, max_rating=5, **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.num_bins = (self.max_rating - self.min_rating) / self.bin_size + 1
        self.bins = tf.range(self.min_rating, self.max_rating + bin_size, bin_size)
        self.bins_01 = tf.range(1, self.num_bins + 1) / self.num_bins

    @staticmethod
    def cdf(x, a, b):
        return tfp.math.betainc(a, b, x)
        # return 1 - (1 - x ** a) ** b

    def call(self, inputs):
        alpha, beta = inputs

        # Calculate cdf at each bin end: (None, num_bins)
        cdf = self.cdf(self.bins_01[:-1], alpha, beta)
        # Replace last cdf bin to avoid issues with calculating d/dx of cdf with convex tails
        cdf = tf.concat([cdf, tf.ones_like(alpha)], axis=-1)
        # Calculate mass in each bin: (None, num_bins)
        mass = tf.concat([cdf[:, :1], tf.experimental.numpy.diff(cdf)], axis=-1)
        # Output tensors: prediction, mass
        return mass

class BetaBinsMassAdaptive(layers.Layer):
    def __init__(self, bin_size=1., min_rating=1, max_rating=5, **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.num_bins = (self.max_rating - self.min_rating) / self.bin_size + 1

    @staticmethod
    def cdf(x, a, b):
        return tfp.math.betainc(a, b, x)

    def call(self, inputs):
        alpha, beta, bins_01 = inputs

        # Calculate cdf at each bin end: (None, num_bins)
        cdf = self.cdf(bins_01[:, :-1], alpha, beta)
        # Replace last cdf bin to avoid issues with calculating d/dx of cdf with convex tails
        cdf = tf.concat([cdf, tf.ones_like(alpha)], axis=-1)
        # Calculate mass in each bin: (None, num_bins)
        mass = tf.concat([cdf[:, :1], tf.experimental.numpy.diff(cdf)], axis=-1)
        # Output tensors: prediction, mass
        return mass

class BetaMean(layers.Layer):
    def __init__(self, bin_size=1., min_rating=1., max_rating=5., **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def call(self, inputs):
        alpha, beta = inputs
        output = (alpha / (alpha + beta))[:, 0] * self.max_rating
        return tf.clip_by_value(output + self.bin_size/2., self.min_rating, self.max_rating)  # (None,)

class BetaMode(layers.Layer):
    def __init__(self, bin_size=1., min_rating=1., max_rating=5., **kwargs):
        super().__init__(**kwargs)
        self.bin_size = tf.cast(bin_size, tf.float32)
        self.min_rating = tf.cast(min_rating, tf.float32)
        self.max_rating = tf.cast(max_rating, tf.float32)

    @tf.function
    def mode(self, alpha, beta):

        a_above_1, b_above_1 = alpha > 1, beta > 1
        a_b = a_above_1 & b_above_1
        a_not_b = a_above_1 & ~b_above_1
        not_a_b = ~a_above_1 & b_above_1
        a_above_b = alpha > beta
        b_above_a = alpha < beta
        return tf.where(
            a_b,
            self._default_mode(alpha, beta),
            tf.where(
                a_not_b,
                self.max_rating,
                tf.where(
                    not_a_b,
                    0.,
                    tf.where(
                        a_above_b,
                        self.max_rating,
                        tf.where(b_above_a,
                                 0.,
                                 0.5*self.max_rating)
                    )
                )
            )
        )

    def _default_mode(self, alpha, beta):
        return ((alpha - 1) / (alpha + beta - 2)) * self.max_rating

    def call(self, inputs):
        alpha, beta = inputs[0][:,0], inputs[1][:,0]
        output = self.mode(alpha, beta)
        return tf.clip_by_value(output + self.bin_size/2., self.min_rating, self.max_rating)  # (None,)

class BetaMedian(layers.Layer):
    def __init__(self, bin_size=1., min_rating=1., max_rating=5., **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

    def call(self, inputs):
        alpha, beta = inputs
        output = tfp.math.betaincinv(alpha, beta, 0.5)[:, 0] * self.max_rating
        return tf.clip_by_value(output + self.bin_size/2., self.min_rating, self.max_rating)  # (None,)

