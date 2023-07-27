import json
import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def process_ml1m(input_data_path, sep="::", names=["uid", "iid", "rating", "t"]):
    df = pd.read_csv(input_data_path, sep=sep, names=names)\
        .drop(names[-1], axis=1)

    # %%
    # Map uid/iid to incremental unique values
    le_uid = LabelEncoder().fit(df["uid"])
    le_iid = LabelEncoder().fit(df["iid"])
    new_old_uid_mapping = np.concatenate([[-1], le_uid.classes_])
    new_old_iid_mapping = np.concatenate([[-1], le_iid.classes_])
    df["uid"] = le_uid.transform(df["uid"]) + 1
    df["iid"] = le_iid.transform(df["iid"]) + 1
    
    data = df.values
    
    metadata = {}
    metadata["num_observed"] = len(data)
    metadata["num_users"] = int(data[:, 0].max()) + 1
    metadata["num_items"] = int(data[:, 1].max()) + 1

    return data, metadata, new_old_uid_mapping, new_old_iid_mapping


def save_data(output_path, data):
    np.save(output_path, data)

def save_metadata(output_path, data: dict):
    with open(output_path, "w") as output_file:
        json.dump(data, output_file)

def main():
    from dotenv import load_dotenv
    load_dotenv()
    project_root = os.environ["PROJECT_ROOT"]

    input_data_path = os.path.join(project_root, "data", "raw", "ml-10m", "ratings.dat")
    output_data_dir = os.path.join(project_root, "data", "preprocessed", "ml-10m", "user_random")
    output_data_path = os.path.join(output_data_dir, "data.npy")
    output_metadata_path = os.path.join(output_data_dir, "metadata.json")
    output_uid_mapping_path = os.path.join(output_data_dir, "new_old_uid_mapping.npy")
    output_iid_mapping_path = os.path.join(output_data_dir, "new_old_iid_mapping.npy")

    os.makedirs(output_data_dir, exist_ok=True)

    data, metadata, new_old_uid_mapping, new_old_iid_mapping = process_ml1m(input_data_path)

    save_data(output_data_path, data)
    save_metadata(output_metadata_path, metadata)
    save_data(output_uid_mapping_path, new_old_uid_mapping)
    save_data(output_iid_mapping_path, new_old_iid_mapping)
        
        
if __name__ == "__main__":
    main()


