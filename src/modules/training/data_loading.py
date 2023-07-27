import glob
import os
import sys

import pandas as pd
import tensorflow as tf

from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = os.environ["PROJECT_ROOT"]
sys.path.append(PROJECT_ROOT)


def _unflatten_tfrecords(record, feature_names=("uid", "iid"), label_name="rating"):
    features = {k:v for k,v in record.items() if k in feature_names}
    labels = record[label_name]
    return features, labels

def _unflatten_csv_dataset():
    pass

def load_data_tfrecord(data_path, batch_size, label_name, shuffle, shuffle_buffer_size, **kwargs):
    # Create a description of the features.
    feature_description = {
        'uid': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'iid': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'rating': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }

    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    filenames = sorted(glob.glob(data_path))
    data = tf.data.TFRecordDataset(filenames).map(_parse_function)\
        .map(lambda x: _unflatten_tfrecords(x, label_name=label_name))
    data = data.batch(batch_size)
    return data

def load_data_csv(data_path, batch_size, label_name, shuffle, shuffle_buffer_size, **kwargs):
    # TODO: Move num epochs to training, not here
    print("SHUFFLE: ", shuffle)
    data = tf.data.experimental.make_csv_dataset(file_pattern=data_path, batch_size=batch_size, label_name=label_name,
                                                 shuffle=shuffle, shuffle_buffer_size=shuffle_buffer_size,
                                                 num_epochs=1)#.cache()
    return data


def _load_data(format, **kwargs):
    if format == "csv":
        print(kwargs)
        return load_data_csv(**kwargs)
    elif format == "tfrecord":
        print(kwargs)
        return load_data_tfrecord(**kwargs)
    else:
        raise NotImplementedError("Currently won't support anything but csv and maybe tfrecords as input format.")

def load_ranking_data_csv(data_path, batch_size=1):
    # Dataset consists of flatten Ragged samples, s.t. the first dimension was `batch_size` but then
    # samples from each RaggedTensor's last dimension were flattened along the first dim. So the
    # first dim is \sum_i^batch_size |I_u|, where |I_u| is the number of interactions from user u
    output_signature = (tf.TensorSpec(shape=(None, 3), dtype=tf.float32))
    paths = sorted(glob.glob(data_path))
    df = pd.concat([pd.read_csv(x) for x in paths])
    user_groups = df.groupby("uid")
    Gen = lambda: (x.values for uid, x in user_groups)
    ds = tf.data.Dataset.from_generator(Gen, output_signature=output_signature) \
        .map(lambda x: ({"uid": tf.cast(x[:, 0], tf.int32), "iid": tf.cast(x[:, 1], tf.int32)}, x[:, 2]))\
        .apply(tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size))\
        .map(lambda x, y: (
            {k: tf.reshape(v, [-1]) for k, v in x.items()},
            tf.reshape(y, [-1]),
            y.row_splits))  # Last tensor allows us to reconstruct the original RaggedTensor during evaluation.
    return ds


# Note that output default is None even if subset not specified
def load_data(data_params):

    output = {"train": None, "validation": None, "test": None}
    for k in output:
        if k in data_params and data_params[k] is not None:
            print(f"LOADING DATA FOR {k} at {data_params[k]}")
            output[k] = _load_data(split=k, **data_params[k], **data_params["general"], **data_params["params"])

    return output