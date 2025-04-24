import pandas as pd
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads JSON data from the given filepath and converts it into a pandas DataFrame.

    Args:
        filepath (str): Path to the JSON file containing raw animal data.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error loading file '{filepath}': {e}")

def preprocess_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Prepares the feature set for clustering, including type conversions and scaling.

    Args:
        df (pd.DataFrame): Original dataframe.
        features (List[str]): List of feature names to use for clustering.

    Returns:
        pd.DataFrame: DataFrame with scaled features.
    """

    df["has_wings"] = df["has_wings"].astype(int)
    df["has_tail"] = df["has_tail"].astype(int)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    df_scaled = pd.DataFrame(scaled_data, columns=features)

    return df_scaled

def cluster_animals (df: pd.DataFrame, scaled_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clusters the animal data using DBSCAN and maps cluster labels to animal names.

    Args:
        df (pd.DataFrame): Original dataframe.
        scaled_data (pd.DataFrame): Scaled features used for clustering.

    Returns:
        pd.DataFrame: DataFrame with cluster labels and mapped animal names.
    """
    dbscan = DBSCAN(eps=0.7, min_samples=5)  
    df["cluster"] = dbscan.fit_predict(scaled_data)

    if df["cluster"].nunique() <= 1 and -1 in df["cluster"].unique():
        print("DBSCAN did not find any valid clusters.")

    cluster_labels = {
        -1: "Outlier",
        0: "Kangaroo",
        1: "Dog",
        2: "Chicken",
        3: "Elephant"
    }

    df["animal"] = df["cluster"].map(cluster_labels)
    df = df[df["cluster"] != -1].copy()

    return df


def save_clusterd_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Saves the clustered and labeled data to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame with clustered data.
        filepath (str): Path to the output CSV file.
    """
    df.to_csv(filepath, index=False)


if __name__ == "__main__":
    RAW_DATA_FILE = "C:/Users/elena/MasterBigData/16_Laboratorio/Python/py_challenge/data/animals.json"
    PROCESSED_DATA_FILE = "C:/Users/elena/MasterBigData/16_Laboratorio/Python/py_challenge/data/animals_clustered.csv"
    features = ["walks_on_n_legs", "height", "weight", "has_wings", "has_tail"]

    try:
        df = load_data(RAW_DATA_FILE)
        scaled = preprocess_features(df, features)
        clustered_df = cluster_animals(df, scaled)
        save_clusterd_data(clustered_df, PROCESSED_DATA_FILE)
        print(f"Clustered data saved to {PROCESSED_DATA_FILE}")
    except Exception as e:
        print(f"Failed to process data: {e}")