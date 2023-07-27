import os

import numpy as np
import pandas as pd
import tensorflow as tf

def get_postprocessed_data_root(project_root, dataset, xval_outer, num_splits_outer, test_pc,
                    xval_inner, num_splits_inner, validation_pc, strategy="user_random", *args, **kwargs):
    # Wandb converts 0. and 1. to 0 and 1: no way of changing it so adjusting here
    validation_pc = int(validation_pc) if validation_pc in [0, 1] else validation_pc
    test_pc = int(test_pc) if test_pc in [0, 1] else test_pc
    return os.path.join(project_root, "data", "postprocessed", dataset, strategy,
                               f"xval_outer={int(xval_outer)};num_outer={num_splits_outer};test_pc={test_pc}",
                               f"xval_inner={int(xval_inner)};num_inner={num_splits_inner};"
                               f"validation_pc={validation_pc}"
                        )

def generate_random_seeds(num_seeds, global_seed=0):
    random_generator = np.random.default_rng(global_seed)
    seeds = random_generator.integers(2**32, size=num_seeds)
    return seeds

def get_shard_size(size, num_shards):
    # Probably doesn't work correctly if size/num_shards is below 1.9
    return int(np.ceil(size/num_shards))


def save_csv(df: pd.DataFrame, output_dir, output_name, num_shards):
    shard_size = get_shard_size(len(df), num_shards)
    start, end = 0, shard_size
    preshard_dir = os.path.join(output_dir, output_name)
    os.makedirs(preshard_dir, exist_ok=True)
    for i in range(num_shards):
        output_path = os.path.join(preshard_dir, f"{i:05}.csv")
        print(f"Writing shard {str(i)} to {output_path}")
        df.iloc[start:end].to_csv(output_path, index=False)
        start, end = np.minimum(start + shard_size, len(df)), np.minimum(end+shard_size, len(df))

def save_tfrecord(df, output_dir, output_name, num_shards):
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def serialize_example(uid, iid, rating):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        # Create a dictionary mapping the feature name to the tf.train.Example-compatible
        # data type.
        feature = {
            'uid': _int64_feature(int(uid)),
            'iid': _int64_feature(int(iid)),
            'rating': _float_feature(rating),
        }

        # Create a Features message using tf.train.Example.

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    shard_size = get_shard_size(len(df), num_shards)
    start, end = 0, shard_size
    preshard_dir = os.path.join(output_dir, output_name)
    os.makedirs(preshard_dir, exist_ok=True)
    for i in range(num_shards):
        output_path = os.path.join(preshard_dir, f"{i:05}.tfrecord")
        with tf.io.TFRecordWriter(output_path) as writer:
            print(f"Writing shard {str(i)} to {output_path}")
            for j in range(start, end):
                example = serialize_example(**df.iloc[j])
                writer.write(example)
        start, end = np.minimum(start + shard_size, len(df)), np.minimum(end + shard_size, len(df))


def save(df, output_dir, output_name, num_shards=1, formats=["csv", "tfrecord"], **kwargs):
    for f in formats:
        if f == "csv":
            csv_kwargs = kwargs["csv"] if "csv" in kwargs else {}
            save_csv(df, output_dir, output_name, num_shards=num_shards, **csv_kwargs)
        elif f == "tfrecord":
            tfrecord_kwargs = kwargs["tfrecord"] if "tfrecord" in kwargs else {}
            save_tfrecord(df, output_dir, output_name, num_shards=num_shards,**tfrecord_kwargs)
        else:
            raise NotImplementedError(f"Cannot save output to format outside {formats}.")