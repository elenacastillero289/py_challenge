import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib 

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads clustered animal data from a CSV file.

    Args:
        filepath (str): Path to the clustered CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(filepath)

def prepare_features_labels(df: pd.DataFrame, feature_cols: list[str], label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates features and labels from the dataset.

    Args:
        df (pd.DataFrame): Input dataset.
        feature_cols (list[str]): Column names to be used as features.
        label_col (str): Column name to be used as the label.

    Returns:
        Tuple of features (X) and labels (y).
    """
    X = df[feature_cols]
    y = df[label_col]
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier on the given data.

    Args:
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.Series): Labels for training.

    Returns:
        Trained RandomForestClassifier model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Evaluates the model and prints accuracy and classification report.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.
    """
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


def save_model(model: RandomForestClassifier, output_path: str) -> None:
    """
    Saves the trained model to a file.

    Args:
        model (RandomForestClassifier): Trained model.
        output_path (str): Path to save the model file.
    """
    joblib.dump(model, output_path)


if __name__ == "__main__":
    DATA_PATH = "C:/Users/elena/MasterBigData/16_Laboratorio/Python/py_challenge/data/animals_clustered.csv"
    MODEL_PATH = "C:/Users/elena/MasterBigData/16_Laboratorio/Python/py_challenge/ml_models/animal_classifier.pkl"
    features = ["walks_on_n_legs", "height", "weight", "has_wings", "has_tail"]
    label = "cluster"  

    df = load_data(DATA_PATH)
    X, y = prepare_features_labels(df, features, label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")