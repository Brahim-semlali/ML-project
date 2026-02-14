"""
Script pour générer toutes les figures du rapport LaTeX dans rapport/figures/
Depuis la racine du projet : python scripts/generate_report_figures.py
"""
import os
import sys

# Racine du projet (parent du dossier scripts/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data, clean_data, scale_features, split_data

FIGURES_DIR = os.path.join(ROOT, 'rapport', 'figures')
DATA_PATH = os.path.join(ROOT, 'dataset', 'winequality-red.csv')

os.makedirs(FIGURES_DIR, exist_ok=True)

print("Chargement des données...")
df = load_data(DATA_PATH)
df = clean_data(df)

X = df.drop(columns=["quality"])
y = df["quality"]
y_binary = (y >= 6).astype(int)

X_scaled, scaler = scale_features(X)
X_train_bin, X_test_bin, y_train_bin, y_test_bin = split_data(X_scaled, y_binary, test_size=0.2, random_state=42)

# === EDA ===
print("Génération EDA...")
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
for i, col in enumerate(X.columns):
    ax = axes.flat[i]
    ax.hist(df[col], bins=20, edgecolor='black', alpha=0.7)
    ax.set_title(col, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_histogrammes.png"), dpi=150)
plt.close()

plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "eda_correlation.png"), dpi=150)
plt.close()

# === PCA ===
print("Génération PCA...")
from sklearn.decomposition import PCA
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)
v2 = pca2.explained_variance_ratio_

plt.figure(figsize=(6, 5))
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=y, cmap="viridis", s=10)
plt.colorbar(label="quality")
plt.title("PCA (2 composantes) – Wine Quality")
plt.xlabel(f"PC1 ({v2[0]:.1%})")
plt.ylabel(f"PC2 ({v2[1]:.1%})")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pca_2d.png"), dpi=150)
plt.close()

pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2], c=y, cmap="viridis", s=10)
plt.colorbar(sc, label="quality")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA (3 composantes)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pca_3d.png"), dpi=150)
plt.close()

# === t-SNE ===
print("Génération t-SNE...")
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="random", learning_rate="auto")
X_tsne = tsne.fit_transform(X_scaled)
plt.figure(figsize=(6, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis", s=10)
plt.colorbar(label="quality")
plt.title("t-SNE – Wine Quality")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "tsne_2d.png"), dpi=150)
plt.close()

# === NMF ===
print("Génération NMF...")
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
X_pos = MinMaxScaler().fit_transform(X)
nmf = NMF(n_components=2, random_state=42)
X_nmf = nmf.fit_transform(X_pos)
plt.figure(figsize=(6, 5))
plt.scatter(X_nmf[:, 0], X_nmf[:, 1], c=y, cmap="viridis", s=10)
plt.colorbar(label="quality")
plt.title("NMF (2 composantes) – Wine Quality")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "nmf_2d.png"), dpi=150)
plt.close()

# === K-Means ===
print("Génération K-Means...")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
pca_vis = PCA(n_components=2)
X_pca_vis = pca_vis.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=4, random_state=42)
labels_k = kmeans.fit_predict(X_pca_vis)
plt.figure(figsize=(6, 5))
plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=labels_k, cmap="tab10", s=10)
plt.title("K-Means (k=4) sur PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "kmeans_clusters.png"), dpi=150)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=labels_k, cmap="tab10", s=10)
axes[0].set_title("Clusters K-Means")
axes[1].scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=y, cmap="viridis", s=10)
axes[1].set_title("Qualité réelle")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "kmeans_comparison.png"), dpi=150)
plt.close()

# === Agglomerative ===
print("Génération Agglomerative...")
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
agg = AgglomerativeClustering(n_clusters=4)
labels_agg = agg.fit_predict(X_pca_vis)
plt.figure(figsize=(6, 5))
plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=labels_agg, cmap="tab10", s=10)
plt.title("Agglomerative Clustering (k=4)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "agg_clusters.png"), dpi=150)
plt.close()

X_sample = X_scaled[:200]
Z = linkage(X_sample, method="ward")
plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90.)
plt.title("Dendrogramme (échantillon 200 vins)")
plt.xlabel("Groupes")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "dendrogram.png"), dpi=150)
plt.close()

# === DBSCAN ===
print("Génération DBSCAN...")
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.7, min_samples=5)
labels_db = dbscan.fit_predict(X_pca_vis)
plt.figure(figsize=(6, 5))
plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=labels_db, cmap="tab10", s=10)
plt.title("DBSCAN sur PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "dbscan_clusters.png"), dpi=150)
plt.close()

outliers = labels_db == -1
plt.figure(figsize=(6, 5))
plt.scatter(X_pca_vis[~outliers, 0], X_pca_vis[~outliers, 1], c=labels_db[~outliers], cmap="tab10", s=10, alpha=0.5)
plt.scatter(X_pca_vis[outliers, 0], X_pca_vis[outliers, 1], c='red', s=30, marker='x', label="Outliers")
plt.title("DBSCAN - Clusters et Outliers")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "dbscan_outliers.png"), dpi=150)
plt.close()

# === Classification ===
print("Génération figures classification...")
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_bin, y_train_bin)
y_pred_lr = lr.predict(X_test_bin)
cm_lr = confusion_matrix(y_test_bin, y_pred_lr)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - Logistic Regression")
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "logistic_confusion_matrix.png"), dpi=150)
plt.close()

# KNN
k_values = [3, 5, 7, 9, 11]
accs, f1s = [], []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_bin, y_train_bin)
    p = knn.predict(X_test_bin)
    accs.append(accuracy_score(y_test_bin, p))
    f1s.append(f1_score(y_test_bin, p))
plt.figure(figsize=(8, 5))
plt.plot(k_values, accs, marker='o', label='Accuracy')
plt.plot(k_values, f1s, marker='s', label='F1-score')
plt.xlabel("Nombre de voisins k")
plt.ylabel("Score")
plt.title("Performance KNN selon k")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "knn_performance.png"), dpi=150)
plt.close()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_bin, y_train_bin)
y_pred_knn = knn.predict(X_test_bin)
cm_knn = confusion_matrix(y_test_bin, y_pred_knn)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - KNN")
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "knn_confusion_matrix.png"), dpi=150)
plt.close()

# Decision Tree
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train_bin, y_train_bin)
y_pred_dt = dt.predict(X_test_bin)
feat_names = X.columns.tolist()
imp = dt.feature_importances_
idx = np.argsort(imp)
plt.figure(figsize=(10, 6))
plt.barh([feat_names[i] for i in idx], imp[idx])
plt.xlabel("Importance")
plt.title("Feature Importance - Decision Tree")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "decisiontree_feature_importance.png"), dpi=150)
plt.close()

cm_dt = confusion_matrix(y_test_bin, y_pred_dt)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - Decision Tree")
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "decisiontree_confusion_matrix.png"), dpi=150)
plt.close()

# SVM
kernels = ['linear', 'rbf', 'poly']
accs_svm, f1s_svm = [], []
for ker in kernels:
    svm = SVC(kernel=ker, probability=True, random_state=42)
    svm.fit(X_train_bin, y_train_bin)
    p = svm.predict(X_test_bin)
    accs_svm.append(accuracy_score(y_test_bin, p))
    f1s_svm.append(f1_score(y_test_bin, p))
plt.figure(figsize=(8, 5))
x_pos = np.arange(len(kernels))
plt.bar(x_pos - 0.2, accs_svm, 0.4, label='Accuracy')
plt.bar(x_pos + 0.2, f1s_svm, 0.4, label='F1-score')
plt.xticks(x_pos, kernels)
plt.ylabel("Score")
plt.title("Performance SVM selon le kernel")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "svm_kernels.png"), dpi=150)
plt.close()

svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train_bin, y_train_bin)
y_pred_svm = svm.predict(X_test_bin)
cm_svm = confusion_matrix(y_test_bin, y_pred_svm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - SVM")
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "svm_confusion_matrix.png"), dpi=150)
plt.close()

# Random Forest
n_est_vals = [50, 100, 200, 300]
accs_rf, f1s_rf = [], []
for n in n_est_vals:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train_bin, y_train_bin)
    p = rf.predict(X_test_bin)
    accs_rf.append(accuracy_score(y_test_bin, p))
    f1s_rf.append(f1_score(y_test_bin, p))
plt.figure(figsize=(8, 5))
plt.plot(n_est_vals, accs_rf, marker='o', label='Accuracy')
plt.plot(n_est_vals, f1s_rf, marker='s', label='F1-score')
plt.xlabel("n_estimators")
plt.ylabel("Score")
plt.title("Performance Random Forest selon n_estimators")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "randomforest_hyperparams.png"), dpi=150)
plt.close()

rf_bin = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
rf_bin.fit(X_train_bin, y_train_bin)
y_pred_rf = rf_bin.predict(X_test_bin)
imp_rf = rf_bin.feature_importances_
idx_rf = np.argsort(imp_rf)
plt.figure(figsize=(10, 6))
plt.barh([feat_names[i] for i in idx_rf], imp_rf[idx_rf])
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest (Binaire)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "randomforest_feature_importance_binary.png"), dpi=150)
plt.close()

cm_rf = confusion_matrix(y_test_bin, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - Random Forest (Binaire)")
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "randomforest_confusion_matrix_binary.png"), dpi=150)
plt.close()

# Random Forest multi-class
X_train_mc, X_test_mc, y_train_mc, y_test_mc = split_data(X_scaled, y, test_size=0.2, random_state=42)
rf_mc = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
rf_mc.fit(X_train_mc, y_train_mc)
y_pred_mc = rf_mc.predict(X_test_mc)
imp_mc = rf_mc.feature_importances_
idx_mc = np.argsort(imp_mc)
plt.figure(figsize=(10, 6))
plt.barh([feat_names[i] for i in idx_mc], imp_mc[idx_mc])
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest (Multi-class)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "randomforest_feature_importance_multiclass.png"), dpi=150)
plt.close()

cm_mc = confusion_matrix(y_test_mc, y_pred_mc)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_mc, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - Random Forest (Multi-class)")
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "randomforest_confusion_matrix_multiclass.png"), dpi=150)
plt.close()

acc_bin = accuracy_score(y_test_bin, y_pred_rf)
f1_bin = f1_score(y_test_bin, y_pred_rf)
acc_mc = accuracy_score(y_test_mc, y_pred_mc)
f1_mc = f1_score(y_test_mc, y_pred_mc, average='macro')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(['Binaire', 'Multi-class'], [acc_bin, acc_mc], color=['blue', 'green'])
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Accuracy: Binaire vs Multi-class")
axes[0].set_ylim([0, 1])
axes[1].bar(['Binaire', 'Multi-class'], [f1_bin, f1_mc], color=['blue', 'green'])
axes[1].set_ylabel("F1-score")
axes[1].set_title("F1-score: Binaire vs Multi-class")
axes[1].set_ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "randomforest_comparison.png"), dpi=150)
plt.close()

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
gb.fit(X_train_bin, y_train_bin)
y_pred_gb = gb.predict(X_test_bin)
imp_gb = gb.feature_importances_
idx_gb = np.argsort(imp_gb)
plt.figure(figsize=(10, 6))
plt.barh([feat_names[i] for i in idx_gb], imp_gb[idx_gb])
plt.xlabel("Importance")
plt.title("Feature Importance - Gradient Boosting")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "gradientboosting_feature_importance.png"), dpi=150)
plt.close()

cm_gb = confusion_matrix(y_test_bin, y_pred_gb)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion - Gradient Boosting")
plt.ylabel("Vraie classe")
plt.xlabel("Classe prédite")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "gradientboosting_confusion_matrix.png"), dpi=150)
plt.close()

# Placeholders MLflow (images à remplacer par des captures d'écran réelles)
fig, ax = plt.subplots(figsize=(10, 4))
ax.text(0.5, 0.5, "Capture d'écran MLflow UI\n(Experiments)\n\nLancer 'mlflow ui' et faire une capture",
        ha='center', va='center', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "mlflow_experiments.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 4))
ax.text(0.5, 0.5, "Capture d'écran MLflow UI\n(Comparaison des modèles)\n\nSélectionner des runs et cliquer sur Compare",
        ha='center', va='center', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "mlflow_comparison.png"), dpi=150)
plt.close()

print(f"Toutes les figures ont été générées dans {FIGURES_DIR}")
print("Vous pouvez maintenant compiler le rapport LaTeX.")
