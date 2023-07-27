import tensorflow as tf


class DiscretizedMSE(tf.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):  # Inputs: (None, 1), (None, num_bins) s.t. last_dim of y_pred is prob_mass
        bin_loss = ((tf.cast(y_true, y_pred.dtype) - 1.) - tf.range(0., 5.)) ** 2  # Calculate the error for each bin
        bin_loss_weighted = bin_loss * y_pred
        return tf.reduce_sum(bin_loss_weighted)


class AdaptedCrossentropy(tf.losses.Loss):
    def __init__(self, bin_size=1., min_rating=1., **kwargs):
        super().__init__()
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.name = "EXTERNAL_XENT_LOSS"
        self.xent_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none", name="INTERNAL_XENT_LOSS", **kwargs)

    def call(self, y_true, y_pred):
        xent = self.xent_loss((y_true - self.min_rating)/self.bin_size, y_pred)
        return tf.reduce_sum(xent)
