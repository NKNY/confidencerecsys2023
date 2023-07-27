import tensorflow as tf

class WeightedMSEWithWeightPenalty(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.loss_fn = lambda x, y: (x-y)**2
        self.reduction = reduction
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)[:,0]
        y_pred, sample_weight = y_pred[:,0], y_pred[:,1]
        pred_loss = sample_weight*self.loss_fn(y_true, y_pred)
        det_loss = tf.math.log(1 / tf.maximum(tf.minimum(sample_weight, 1e6), 1e-6))
        sample_loss = pred_loss + det_loss
        if self.reduction == tf.keras.losses.Reduction.NONE:
            return sample_loss
        else:
            if self.reduction == tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE:
                return tf.reduce_mean(sample_loss)
            else:
                return tf.reduce_sum(sample_loss)

class CustomRMSE(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.loss_fn = lambda x, y: (x-y)**2
        self.reduction = reduction

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        if len(y_pred.shape) > len(y_true.shape):
            y_pred = y_pred[:,0]
        sample_loss = (y_true-y_pred)**2
        if self.reduction == tf.keras.losses.Reduction.NONE:
            return sample_loss
        else:
            if self.reduction == tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE:
                return tf.reduce_mean(sample_loss)
            else:
                return tf.reduce_sum(sample_loss)