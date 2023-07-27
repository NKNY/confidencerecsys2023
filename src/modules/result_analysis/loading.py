import pickle

import pandas as pd

# How to preprocess special columns
process_map = {
    "bins_mass": lambda x: {f"bins_mass_{i}": (col.T) for i, col in enumerate(x.T)},
    "X": lambda x: {k: v for k, v in x.items()}
}


def path_to_df(path):
    # Load raw data with model predictions
    with open(path, "rb") as input_file:
        raw_data = pickle.load(input_file)
    return raw_to_df(raw_data)


def raw_to_df(raw_data):
    data = raw_to_dict_of_lists(raw_data)
    # Save as pandas DataFrame
    df = pd.DataFrame.from_dict(data)
    return df[["uid", "iid", "rating"] + [x for x in df.columns if x not in ["uid", "iid", "rating"]]]


def raw_to_dict_of_lists(raw_data):
    # Things like bins_mass or X are split into separate columns with one value per columns
    data = {}
    for k, v in raw_data.items():
        if k not in process_map:
            update = {}
            if len(v.shape) == 2:
                if v.shape[1] == 1:
                    update[k] = v.flatten()
                else:
                    for col in range(v.shape[1]):
                        update[f"{k}_{col}"] = v[:,col].flatten()
            else:
                update[k] = v.flatten()
        else:
            update = process_map[k](v)
        data = {**data, **update}
    return data