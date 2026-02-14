import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Charger le dataset Wine Quality depuis un chemin donné."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyer les données : gestion des valeurs manquantes, doublons, etc.
    Pour ce dataset, il n'y a normalement pas de valeurs manquantes,
    mais on supprime par sécurité les éventuelles lignes invalides.
    """
    df_clean = df.copy()
    df_clean = df_clean.dropna().drop_duplicates()
    return df_clean


def scale_features(X):
    """
    Appliquer une normalisation standard (moyenne 0, variance 1)
    et retourner X_scaled ainsi que le scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Diviser les données en ensembles d'entraînement et de test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

