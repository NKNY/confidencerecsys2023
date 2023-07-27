import tensorflow as tf

class AdaptedCrossentropy(tf.keras.metrics.Metric):
    def __init__(self, bin_size=1., min_rating=1., **kwargs):
        super().__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.xent_fn = tf.keras.metrics.sparse_categorical_crossentropy
        self.xent = self.add_weight("xent", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        update_per_sample = self.xent_fn(y_true-tf.ones_like(y_true), y_pred)

        if sample_weight is not None:
            update_per_sample = tf.multiply(update_per_sample, sample_weight)
        self.xent.assign_add(tf.reduce_sum(update_per_sample))

    def result(self):
        return self.xent


class CustomRMSEMetric(tf.keras.metrics.Metric):
    def __init__(self, min_rating=1., max_rating=5., name="root_mean_squared_error", **kwargs):
        super().__init__(name=name, **kwargs)
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.num = self.add_weight("AdaptedAccuracyNum", initializer="zeros")
        self.denom = self.add_weight("AdaptedAccuracyDenom", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Ensure same types
        y_true = tf.cast(y_true, self.num.dtype)
        y_pred = tf.cast(y_pred, self.num.dtype)
        # Ensure y_pred in allowed range
        y_pred = tf.clip_by_value(y_pred, self.min_rating, self.max_rating)
        # Ensure both flat
        y_true, y_pred = tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])
        error_sq = (y_true - y_pred)**2
        self.num.assign_add(tf.reduce_sum(error_sq))
        self.denom.assign_add(tf.cast(tf.shape(y_pred)[0], y_pred.dtype))

    def result(self):
        return tf.math.sqrt(self.num/self.denom)

class AdaptedAccuracy(tf.keras.metrics.Metric):
    def __init__(self,  bin_size=1., min_rating=0.5, max_rating=5., **kwargs):
        super(AdaptedAccuracy, self).__init__(**kwargs)
        self.bin_size = bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.num = self.add_weight("AdaptedAccuracyNum", initializer="zeros")
        self.denom = self.add_weight("AdaptedAccuracyDenom", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        success = tf.math.abs(tf.cast(y_true, y_pred.dtype)/self.bin_size
                              - tf.clip_by_value(y_pred, self.min_rating, self.max_rating)/self.bin_size) < 0.5
        self.num.assign_add(tf.reduce_sum(tf.cast(success, y_pred.dtype)))
        self.denom.assign_add(tf.cast(tf.shape(y_pred)[0], y_pred.dtype))

    def result(self):
        return self.num/self.denom
