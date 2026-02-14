"""
Génère les figures manquantes du rapport (sans MLflow).
Depuis la racine du projet : python scripts/generate_figures.py
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data, clean_data, scale_features, split_data

FIG_DIR = os.path.join(ROOT, 'rapport', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
DATA_PATH = os.path.join(ROOT, "dataset", "winequality-red.csv")


def _load_wine():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    X = df.drop(columns=["quality"])
    y = df["quality"]
    X_scaled, _ = scale_features(X)
    return X, X_scaled, y, df


def generate_naivebayes():
    """Figure 16 – Naive Bayes - Matrice de confusion et courbe ROC"""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

    _, X_scaled, _, _ = _load_wine()
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df["quality_binary"] = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality", "quality_binary"])
    y = df["quality_binary"]
    X_scaled, _ = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=0.2, random_state=42)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    y_proba = nb.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Matrice de confusion - Naive Bayes')
    axes[0].set_ylabel('Vraie classe')
    axes[0].set_xlabel('Classe prédite')
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, label=f'Naive Bayes (AUC={auc_score:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('Taux faux positifs')
    axes[1].set_ylabel('Taux vrais positifs')
    axes[1].set_title('Courbe ROC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'naivebayes_confusion_matrix.png'))
    plt.close()
    print("Généré : naivebayes_confusion_matrix.png")


def generate_agg_comparison():
    """Figure 10 – Comparaison clusters agglomératifs vs qualité réelle"""
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering

    X, X_scaled, y, _ = _load_wine()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
    labels = agg.fit_predict(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10)
    axes[0].set_title('Clusters Agglomerative (k=4)')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=10)
    axes[1].set_title('Qualité réelle')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'agg_comparison.png'))
    plt.close()
    print("Généré : agg_comparison.png")


def generate_dbscan_comparison():
    """Figure 12 – Comparaison DBSCAN vs qualité réelle"""
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN

    X, X_scaled, y, _ = _load_wine()
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    dbscan = DBSCAN(eps=0.7, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10)
    axes[0].set_title('Clusters DBSCAN')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=10)
    axes[1].set_title('Qualité réelle')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'dbscan_comparison.png'))
    plt.close()
    print("Généré : dbscan_comparison.png")


def generate_gmm_bic():
    """Figure 13 – BIC en fonction du nombre de composantes GMM"""
    from sklearn.mixture import GaussianMixture

    _, X_scaled, _, _ = _load_wine()
    n_components_range = range(2, 8)
    bics = []
    for k in n_components_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X_scaled)
        bics.append(gmm.bic(X_scaled))

    plt.figure(figsize=(7, 4))
    plt.plot(list(n_components_range), bics, 'bo-')
    plt.xlabel('Nombre de composantes')
    plt.ylabel('BIC')
    plt.title('BIC pour choisir le nombre de composantes GMM')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'gmm_bic.png'))
    plt.close()
    print("Généré : gmm_bic.png")


def generate_decisiontree_tree():
    """Figure 18 – Structure de l'arbre de décision"""
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df["quality_binary"] = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality", "quality_binary"])
    y = df["quality_binary"]
    feature_names = X.columns.tolist()
    X_scaled, _ = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=0.2, random_state=42)

    tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_clf.fit(X_train, y_train)

    plt.figure(figsize=(20, 10))
    plot_tree(tree_clf, feature_names=feature_names, class_names=['Mauvais/Moyen', 'Bon'],
              filled=True, max_depth=3, fontsize=10)
    plt.title("Decision Tree (premiers niveaux)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'decisiontree_tree.png'))
    plt.close()
    print("Généré : decisiontree_tree.png")


def main():
    print("Génération des figures manquantes du rapport...\n")
    generate_naivebayes()
    generate_agg_comparison()
    generate_dbscan_comparison()
    generate_gmm_bic()
    generate_decisiontree_tree()
    print("\nTerminé. Figures enregistrées dans rapport/figures/")
    print("\nPour les figures 27 et 28 (MLflow) :")
    print("  1. Lancer MLflow : mlflow ui --backend-store-uri sqlite:///classification/mlflow.db")
    print("  2. Ouvrir http://127.0.0.1:5000")
    print("  3. Faire des captures d'écran et les enregistrer sous :")
    print("     - mlflow_experiments.png")
    print("     - mlflow_comparison.png")
    print("     dans le dossier rapport/figures/")


if __name__ == "__main__":
    main()
