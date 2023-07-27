# Copyright 2022 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keras metrics in TF-Ranking.
NOTE: For metrics that compute a ranking, ties are broken randomly. This means
that metrics may be stochastic if items with equal scores are provided.
WARNING: Some metrics (e.g. Recall or MRR) are not well-defined when there are
no relevant items (e.g. if `y_true` has a row of only zeroes). For these cases,
the TF-Ranking metrics will evaluate to `0`.
"""

import tensorflow as tf
import src.modules.metrics.ranking.metrics_impl as metrics_impl
import src.modules.metrics.ranking.keras_utils as keras_utils
import src.modules.metrics.ranking.utils as utils

class _RankingMetric(tf.keras.metrics.Mean):
  """Implements base ranking metric class.
  Please see tf.keras.metrics.Mean for more information about such a class and
  https://www.tensorflow.org/tutorials/distribute/custom_training on how to do
  customized training.
  """

  def __init__(self, name=None, dtype=None, ragged=False, **kwargs):
    super(_RankingMetric, self).__init__(name=name, dtype=dtype, **kwargs)
    # An instance of `metrics_impl._RankingMetric`.
    # Overwrite this in subclasses.
    self._metric = None
    self._ragged = ragged

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates metric statistics.
    `y_true` and `y_pred` should have the same shape.
    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.
    Returns:
      Update op.
    """
    y_true = tf.cast(y_true, self._dtype)
    y_pred = tf.cast(y_pred, self._dtype)

    per_list_metric_val, per_list_metric_weights = self._metric.compute(
        y_true, y_pred, sample_weight)
    return super(_RankingMetric, self).update_state(
        per_list_metric_val, sample_weight=per_list_metric_weights)

  def get_config(self):
    config = super(_RankingMetric, self).get_config()
    config.update({
        "ragged": self._ragged,
    })
    return config

class NDCGMetric(_RankingMetric):
  r"""Normalized discounted cumulative gain (NDCG).
  Normalized discounted cumulative gain ([Järvelin et al, 2002][jarvelin2002])
  is the normalized version of `tfr.keras.metrics.DCGMetric`.
  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:
  ```
  NDCG(y, s) = DCG(y, s) / DCG(y, y)
  DCG(y, s) = sum_i gain(y_i) * rank_discount(rank(s_i))
  ```
  NOTE: The `gain_fn` and `rank_discount_fn` should be keras serializable.
  Please see `tfr.keras.utils.pow_minus_1` and `tfr.keras.utils.log2_inverse` as
  examples when defining user customized functions.
  Standalone usage:
  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> ndcg = tfr.keras.metrics.NDCGMetric()
  >>> ndcg(y_true, y_pred).numpy()
  0.6934264
  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> ndcg = tfr.keras.metrics.NDCGMetric(ragged=True)
  >>> ndcg(y_true, y_pred).numpy()
  0.7974351
  Usage with the `compile()` API:
  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.NDCGMetric()])
  ```
  Definition:
  $$
  \text{NDCG}(\{y\}, \{s\}) =
  \frac{\text{DCG}(\{y\}, \{s\})}{\text{DCG}(\{y\}, \{y\})} \\
  \text{DCG}(\{y\}, \{s\}) =
  \sum_i \text{gain}(y_i) \cdot \text{rank_discount}(\text{rank}(s_i))
  $$
  where $\text{rank}(s_i)$ is the rank of item $i$ after sorting by scores
  $s$ with ties broken randomly.
  References:
    - [Cumulated gain-based evaluation of IR techniques, Järvelin et al,
       2002][jarvelin2002]
  [jarvelin2002]: https://dl.acm.org/doi/10.1145/582415.582418
  """

  def __init__(self,
               name=None,
               topn=None,
               gain_fn=None,
               rank_discount_fn=None,
               dtype=None,
               ragged=True,
               **kwargs):
    super(NDCGMetric, self).__init__(name=name, dtype=dtype, ragged=ragged,
                                     **kwargs)

    self._topn = topn
    self._gain_fn = utils.eval_if_str(gain_fn) or keras_utils.pow_minus_1
    self._rank_discount_fn = utils.eval_if_str(rank_discount_fn) or keras_utils.log2_inverse
    self._metric = metrics_impl.NDCGMetric(
        name=name,
        topn=topn,
        gain_fn=self._gain_fn,
        rank_discount_fn=self._rank_discount_fn,
        ragged=ragged)

  def get_config(self):
    base_config = super(NDCGMetric, self).get_config()
    config = {
        "topn": self._topn,
        "gain_fn": self._gain_fn,
        "rank_discount_fn": self._rank_discount_fn,
    }
    config.update(base_config)
    return config


class DCGMetric(_RankingMetric):
  r"""Discounted cumulative gain (DCG).
  Discounted cumulative gain ([Järvelin et al, 2002][jarvelin2002]).
  For each list of scores `s` in `y_pred` and list of labels `y` in `y_true`:
  ```
  DCG(y, s) = sum_i gain(y_i) * rank_discount(rank(s_i))
  ```
  NOTE: The `gain_fn` and `rank_discount_fn` should be keras serializable.
  Please see `tfr.keras.utils.pow_minus_1` and `tfr.keras.utils.log2_inverse` as
  examples when defining user customized functions.
  Standalone usage:
  >>> y_true = [[0., 1., 1.]]
  >>> y_pred = [[3., 1., 2.]]
  >>> dcg = tfr.keras.metrics.DCGMetric()
  >>> dcg(y_true, y_pred).numpy()
  1.1309297
  >>> # Using ragged tensors
  >>> y_true = tf.ragged.constant([[0., 1.], [1., 2., 0.]])
  >>> y_pred = tf.ragged.constant([[2., 1.], [2., 5., 4.]])
  >>> dcg = tfr.keras.metrics.DCGMetric(ragged=True)
  >>> dcg(y_true, y_pred).numpy()
  2.065465
  Usage with the `compile()` API:
  ```python
  model.compile(optimizer='sgd', metrics=[tfr.keras.metrics.DCGMetric()])
  ```
  Definition:
  $$
  \text{DCG}(\{y\}, \{s\}) =
  \sum_i \text{gain}(y_i) \cdot \text{rank_discount}(\text{rank}(s_i))
  $$
  where $\text{rank}(s_i)$ is the rank of item $i$ after sorting by scores
  $s$ with ties broken randomly.
  References:
    - [Cumulated gain-based evaluation of IR techniques, Järvelin et al,
       2002][jarvelin2002]
  [jarvelin2002]: https://dl.acm.org/doi/10.1145/582415.582418
  """

  def __init__(self,
               name=None,
               topn=None,
               gain_fn=None,
               rank_discount_fn=None,
               dtype=None,
               ragged=True,
               **kwargs):
    super(DCGMetric, self).__init__(name=name, dtype=dtype, ragged=ragged,
                                    **kwargs)
    self._topn = topn
    self._gain_fn = utils.eval_if_str(gain_fn) or keras_utils.pow_minus_1
    self._rank_discount_fn = utils.eval_if_str(rank_discount_fn) or keras_utils.log2_inverse
    self._metric = metrics_impl.DCGMetric(
        name=name,
        topn=topn,
        gain_fn=self._gain_fn,
        rank_discount_fn=self._rank_discount_fn,
        ragged=ragged)

  def get_config(self):
    base_config = super(DCGMetric, self).get_config()
    config = {
        "topn": self._topn,
        "gain_fn": self._gain_fn,
        "rank_discount_fn": self._rank_discount_fn,
    }
    config.update(base_config)
    return config