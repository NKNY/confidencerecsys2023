import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.special as ssp

bins_mass_cols = [f"bins_mass_{i}" for i in range(10)]
standardised_cols = ["uid", "iid", "rating", "mean"] + bins_mass_cols + ["var", "err_mean", "p_highest", "highest_bin", "closest_bin", "d_closest", "correct_bin", "highest_correct"]
def standardise_LBD(df, cols_to_keep = ["alpha", "beta", "mu", "upsilon"]):
    df["var"] = df["alpha"]*df["beta"]/((df["alpha"]+df["beta"])**2 * (df["alpha"]+df["beta"]+1))*25
    df["err_mean"] = df["rating"] - df["mean"]
    df["p_highest"] = np.max(df[bins_mass_cols].values, axis=-1)
    df["highest_bin"] = np.argmax(df[bins_mass_cols].values, axis=-1)
    df["closest_bin"] = (df["mean"] * 2).round()/2
    df["d_closest"] = np.abs(df["mean"] - df["closest_bin"])
    df["correct_bin"] = df["closest_bin"] == df["rating"]
    df["highest_correct"] = df["highest_bin"] == (df["rating"]*2-1)

    df = df[standardised_cols + cols_to_keep]
    return df

def standardise_adaptive_LBD(df, cols_to_keep = ["alpha", "beta", "mu", "upsilon"]):
    df["mean"] = df["bins_mean"]
    df["var"] = np.sum((np.arange(0.5, 5.5, 0.5)**2)*df[bins_mass_cols].values, axis=-1) - df["mean"]**2
    df["err_mean"] = df["rating"] - df["mean"]
    df["p_highest"] = np.max(df[bins_mass_cols].values, axis=-1)
    df["highest_bin"] = np.argmax(df[bins_mass_cols].values, axis=-1)
    df["closest_bin"] = (df["mean"] * 2).round()/2
    df["d_closest"] = np.abs(df["mean"] - df["closest_bin"])
    df["correct_bin"] = df["closest_bin"] == df["rating"]
    df["highest_correct"] = df["highest_bin"] == (df["rating"]*2-1)

    df = df[standardised_cols + cols_to_keep]
    return df

def standardise_ordrec(df, cols_to_keep = ["pred"]):
    df["mean"] = np.clip(df["bins_mean"], 0.5, 5)
    df["pred"] = df["mean"]
    df["var"] = np.var(df[bins_mass_cols].values, axis=-1)
    df["var"] = np.sum((np.arange(0.5, 5.5, 0.5)**2)*df[bins_mass_cols].values, axis=-1) - df["mean"]**2
    df["err_mean"] = df["rating"] - df["mean"]
    df["p_highest"] = np.max(df[bins_mass_cols].values, axis=-1)
    df["highest_bin"] = np.argmax(df[bins_mass_cols].values, axis=-1)
    df["closest_bin"] = (df["mean"] * 2).round()/2
    df["d_closest"] = np.abs(df["mean"] - df["closest_bin"]) 
    df["correct_bin"] = df["closest_bin"] == df["rating"]
    df["highest_correct"] = df["highest_bin"] == (df["rating"]*2-1)
    
    return df[standardised_cols + cols_to_keep]

def standardise_cmf(df, cols_to_keep = ["pred"]):
    df["mean"] = np.clip(df["pred"], 0.5, 5)
    df["var"] = 1/df["y_pred_1"]
    df["err_mean"] = df["rating"] - df["mean"]
    norm = ss.norm(df["mean"], np.sqrt(df["var"]))
    norm_cdf = norm.cdf((np.arange(0.5, 5.5, 0.5) + 0.25)[:,None]).T
    norm_cdf[:,-1] = 1.
    norm_cdf = np.concatenate([np.zeros((len(norm_cdf), 1)), norm_cdf], axis=-1)
    norm_bins = norm_cdf[:,1:]-norm_cdf[:,:-1]
    df = pd.concat([df, pd.DataFrame(norm_bins, columns=bins_mass_cols)], axis=1)
    df["p_highest"] = np.max(df[bins_mass_cols].values, axis=-1)
    df["highest_bin"] = np.argmax(df[bins_mass_cols].values, axis=-1)
    df["closest_bin"] = (df["mean"] * 2).round()/2
    df["d_closest"] = np.abs(df["mean"] - df["closest_bin"]) 
    df["correct_bin"] = df["closest_bin"] == df["rating"]
    df["highest_correct"] = df["highest_bin"] == (df["rating"]*2-1)
    
    return df[standardised_cols + cols_to_keep]

def standardise_mf(df, cols_to_keep = [], l2=0.037438325237344985):
    df["mean"] = np.clip(df["pred"], 0.5, 5)
    df["var"] = 0.01 / l2
    df["err_mean"] = df["rating"] - df["mean"]
    norm = ss.norm(df["mean"], np.sqrt(df["var"]))
    norm_cdf = norm.cdf((np.arange(0.5, 5.5, 0.5) + 0.25)[:,None]).T
    norm_cdf[:,-1] = 1.
    norm_cdf = np.concatenate([np.zeros((len(norm_cdf), 1)), norm_cdf], axis=-1)
    norm_bins = norm_cdf[:,1:]-norm_cdf[:,:-1]
    df = pd.concat([df, pd.DataFrame(norm_bins, columns=bins_mass_cols)], axis=1)
    df["p_highest"] = np.max(df[bins_mass_cols].values, axis=-1)
    df["highest_bin"] = np.argmax(df[bins_mass_cols].values, axis=-1)
    df["closest_bin"] = (df["mean"] * 2).round()/2
    df["d_closest"] = np.abs(df["mean"] - df["closest_bin"]) 
    df["correct_bin"] = df["closest_bin"] == df["rating"]
    df["highest_correct"] = df["highest_bin"] == (df["rating"]*2-1)
    return df[standardised_cols + cols_to_keep]


def standardise_model(k, df):
    standardise_fn = {"LBDA": standardise_adaptive_LBD, "LBD": standardise_LBD, "CMF": standardise_cmf, "MF": standardise_mf, "OrdRec": standardise_ordrec}
    for name, fn in standardise_fn.items():
        if k.startswith(name):
            return(fn(df))