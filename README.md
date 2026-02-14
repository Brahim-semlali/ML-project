# Projet Machine Learning – Wine Quality

**Dépôt GitHub :** [https://github.com/Brahim-semlali/ML-project](https://github.com/Brahim-semlali/ML-project)

Ce projet implémente le cahier des charges sur le dataset **Wine Quality (vin rouge)** :

- Chargement et exploration des données (EDA)
- Pré-traitement et préparation des features
- Réduction de dimension : **PCA**, **t-SNE**, **NMF**, **LDA**
- Clustering : **K-Means**, **Agglomerative**, **DBSCAN**, **GMM**
- Classification : **Logistic Regression, KNN, Decision Tree, SVM, Random Forest, Gradient Boosting, AdaBoost, Naive Bayes, Neural Network**
- Suivi des expériences avec **MLflow**

Toutes les étapes sont également regroupées dans un notebook pipeline : `wine_quality_ml.ipynb`.

---

## Structure du projet

Chaque **algorithme** a son propre dossier : **notebook + images** au même endroit (ex. `clustering/KMeans/` contient `KMeans.ipynb` et les figures générées).

```
ML-project/
├── dataset/
│   └── winequality-red.csv
├── src/
│   └── preprocessing.py
├── reduction/               # Réduction de dimension
│   ├── PCA/
│   │   ├── PCA.ipynb
│   │   ├── pca_2d.png
│   │   └── pca_3d.png
│   ├── tSNE/
│   │   ├── tSNE.ipynb
│   │   └── tsne_2d.png
│   ├── NMF/
│   │   ├── NMF.ipynb
│   │   └── nmf_2d.png
│   └── LDA/
│       ├── LDA.ipynb
│       └── lda_2d.png
├── clustering/              # Clustering
│   ├── KMeans/
│   │   ├── KMeans.ipynb
│   │   ├── kmeans_clusters.png
│   │   └── kmeans_comparison.png
│   ├── AgglomerativeClustering/
│   │   ├── AgglomerativeClustering.ipynb
│   │   ├── agg_clusters.png
│   │   ├── agg_comparison.png
│   │   └── dendrogram.png
│   ├── DBSCAN/
│   │   ├── DBSCAN.ipynb
│   │   └── dbscan_*.png
│   └── GMM/
│       ├── GMM.ipynb
│       ├── gmm_bic.png
│       └── gmm_clusters.png
├── classification/          # Classification (même principe : Algo/Algo.ipynb + *.png)
│   ├── LogisticRegression/
│   ├── KNN/
│   ├── DecisionTree/
│   ├── SVM/
│   ├── RandomForest/
│   ├── GradientBoosting/
│   ├── AdaBoost/
│   ├── NaiveBayes/
│   └── NeuralNetwork/
├── rapport/
│   ├── rapport_wine_quality.tex
│   ├── figures/             # Copie des figures (pour le PDF)
│   ├── build.ps1
│   └── build.bat
├── scripts/
│   ├── generate_figures.py          # Génère quelques figures manquantes
│   ├── generate_report_figures.py   # Génère toutes les figures
│   └── collect_figures.py           # Copie figures des dossiers algo → rapport/figures/
├── wine_quality_ml.ipynb
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Compiler le rapport PDF

**Important :** La compilation doit se faire **depuis le dossier `rapport/`** (ou via les scripts fournis).

- **Depuis la racine du projet (PowerShell) :**
  ```powershell
  .\rapport\build.ps1
  ```
- **Depuis le dossier rapport :**
  ```powershell
  cd rapport
  .\build.ps1
  ```
  ou sous CMD :
  ```cmd
  cd rapport
  build.bat
  ```
- **À la main :**
  ```powershell
  cd rapport
  pdflatex rapport_wine_quality.tex
  pdflatex rapport_wine_quality.tex
  ```

Le PDF généré est `rapport/rapport_wine_quality.pdf`.

---

## Générer / mettre à jour les figures du rapport

Depuis la **racine du projet** :

- **Copier les figures** des dossiers algorithmes vers `rapport/figures/` (après avoir exécuté les notebooks) :
  ```bash
  python scripts/collect_figures.py
  ```
- **Générer quelques figures** sans exécuter les notebooks (Naive Bayes, agg vs qualité, DBSCAN vs qualité, GMM BIC, arbre de décision) :
  ```bash
  python scripts/generate_figures.py
  ```
- **Générer toutes les figures** du rapport (EDA, PCA, clustering, classification, etc.) :
  ```bash
  python scripts/generate_report_figures.py
  ```

Les figures des notebooks sont **dans le dossier de chaque algorithme** ; `collect_figures.py` les recopie dans `rapport/figures/` pour le PDF.

---

## Organisation des notebooks (par algorithme)

- **Réduction** : `reduction/PCA/`, `reduction/tSNE/`, `reduction/NMF/`, `reduction/LDA/` — chaque dossier contient le notebook et ses images.
- **Clustering** : `clustering/KMeans/`, `clustering/AgglomerativeClustering/`, `clustering/DBSCAN/`, `clustering/GMM/`.
- **Classification** : `classification/LogisticRegression/`, `classification/KNN/`, … jusqu’à `classification/NeuralNetwork/`.
- **Pipeline complet** : `wine_quality_ml.ipynb` (à la racine).

Chaque notebook détecte la **racine du projet** (dossier contenant `src/` et `dataset/`) et charge les données depuis `dataset/winequality-red.csv`. Les figures sont enregistrées **dans le dossier de l’algorithme**.

