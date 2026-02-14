# Rapport LaTeX - Projet Wine Quality

**Projet (code + données) :** [https://github.com/Brahim-semlali/ML-project](https://github.com/Brahim-semlali/ML-project)

Ce dossier contient le rapport LaTeX du projet Machine Learning sur le dataset Wine Quality. Sur le dépôt GitHub sont versionnés le **PDF** (`rapport_wine_quality.pdf`) et le dossier **figures/** ; les fichiers sources `.tex` et auxiliaires ne sont pas poussés.

## Structure

```
rapport/
├── rapport_wine_quality.tex    # Fichier LaTeX principal
├── figures/                     # Dossier pour les figures/screenshots
│   ├── eda_histogrammes.png
│   ├── eda_correlation.png
│   ├── pca_2d.png
│   ├── pca_3d.png
│   ├── tsne_2d.png
│   ├── nmf_2d.png
│   ├── kmeans_clusters.png
│   ├── kmeans_comparison.png
│   ├── agg_clusters.png
│   ├── dendrogram.png
│   ├── dbscan_clusters.png
│   ├── dbscan_outliers.png
│   ├── logistic_confusion_matrix.png
│   ├── knn_performance.png
│   ├── knn_confusion_matrix.png
│   ├── decisiontree_feature_importance.png
│   ├── decisiontree_confusion_matrix.png
│   ├── svm_kernels.png
│   ├── svm_confusion_matrix.png
│   ├── randomforest_hyperparams.png
│   ├── randomforest_feature_importance_binary.png
│   ├── randomforest_confusion_matrix_binary.png
│   ├── randomforest_feature_importance_multiclass.png
│   ├── randomforest_confusion_matrix_multiclass.png
│   ├── randomforest_comparison.png
│   ├── gradientboosting_feature_importance.png
│   ├── gradientboosting_confusion_matrix.png
│   ├── mlflow_experiments.png
│   └── mlflow_comparison.png
└── README.md                    # Ce fichier
```

## Comment compiler le rapport

### Option 1 : Avec pdflatex (recommandé)

```bash
cd rapport
pdflatex rapport_wine_quality.tex
pdflatex rapport_wine_quality.tex  # Deux fois pour les références
```

### Option 2 : Avec un éditeur LaTeX

- Ouvrir `rapport_wine_quality.tex` dans :
  - **Overleaf** (en ligne) : https://www.overleaf.com/
  - **TeXstudio** (local)
  - **TeXmaker** (local)
  - **VS Code** avec extension LaTeX Workshop

### Option 3 : Avec latexmk (automatique)

```bash
cd rapport
latexmk -pdf rapport_wine_quality.tex
```

## Où trouver les figures

### 1. Figures EDA (Exploration des données)

Depuis `wine_quality_ml.ipynb` :
- **Histogrammes** : Exécuter la cellule avec `df.hist(...)` → Screenshot → Sauvegarder comme `eda_histogrammes.png`
- **Heatmap corrélation** : Exécuter la cellule avec `sns.heatmap(...)` → Screenshot → Sauvegarder comme `eda_correlation.png`

### 2. Figures Réduction de dimension

- **PCA 2D** : Depuis `reduction/PCA.ipynb` → Screenshot de la figure PCA 2D → `pca_2d.png`
- **PCA 3D** : Depuis `reduction/PCA.ipynb` → Screenshot de la figure PCA 3D → `pca_3d.png`
- **t-SNE** : Depuis `reduction/tSNE.ipynb` → Screenshot → `tsne_2d.png`
- **NMF** : Depuis `reduction/NMF.ipynb` → Screenshot → `nmf_2d.png`

### 3. Figures Clustering

- **K-Means clusters** : Depuis `clustering/KMeans.ipynb` → Screenshot → `kmeans_clusters.png`
- **K-Means comparison** : Depuis `clustering/KMeans.ipynb` → Screenshot comparaison → `kmeans_comparison.png`
- **Agglomerative clusters** : Depuis `clustering/AgglomerativeClustering.ipynb` → Screenshot → `agg_clusters.png`
- **Dendrogramme** : Depuis `clustering/AgglomerativeClustering.ipynb` → Screenshot → `dendrogram.png`
- **DBSCAN clusters** : Depuis `clustering/DBSCAN.ipynb` → Screenshot → `dbscan_clusters.png`
- **DBSCAN outliers** : Depuis `clustering/DBSCAN.ipynb` → Screenshot → `dbscan_outliers.png`

### 4. Figures Classification

- **Logistic Regression CM** : Depuis `classification/LogisticRegression.ipynb` → Screenshot → `logistic_confusion_matrix.png`
- **KNN performance** : Depuis `classification/KNN.ipynb` → Screenshot courbe performance → `knn_performance.png`
- **KNN CM** : Depuis `classification/KNN.ipynb` → Screenshot → `knn_confusion_matrix.png`
- **Decision Tree importance** : Depuis `classification/DecisionTree.ipynb` → Screenshot → `decisiontree_feature_importance.png`
- **Decision Tree CM** : Depuis `classification/DecisionTree.ipynb` → Screenshot → `decisiontree_confusion_matrix.png`
- **SVM kernels** : Depuis `classification/SVM.ipynb` → Screenshot → `svm_kernels.png`
- **SVM CM** : Depuis `classification/SVM.ipynb` → Screenshot → `svm_confusion_matrix.png`
- **Random Forest hyperparams** : Depuis `classification/RandomForest.ipynb` → Screenshot → `randomforest_hyperparams.png`
- **Random Forest importance (binaire)** : Depuis `classification/RandomForest.ipynb` → Screenshot → `randomforest_feature_importance_binary.png`
- **Random Forest CM (binaire)** : Depuis `classification/RandomForest.ipynb` → Screenshot → `randomforest_confusion_matrix_binary.png`
- **Random Forest importance (multi-class)** : Depuis `classification/RandomForest.ipynb` → Screenshot → `randomforest_feature_importance_multiclass.png`
- **Random Forest CM (multi-class)** : Depuis `classification/RandomForest.ipynb` → Screenshot → `randomforest_confusion_matrix_multiclass.png`
- **Random Forest comparison** : Depuis `classification/RandomForest.ipynb` → Screenshot → `randomforest_comparison.png`
- **Gradient Boosting importance** : Depuis `classification/GradientBoosting.ipynb` → Screenshot → `gradientboosting_feature_importance.png`
- **Gradient Boosting CM** : Depuis `classification/GradientBoosting.ipynb` → Screenshot → `gradientboosting_confusion_matrix.png`

### 5. Figures MLflow

- **MLflow experiments** : Screenshot de l'interface MLflow (`http://127.0.0.1:5000`) → Vue Experiments → `mlflow_experiments.png`
- **MLflow comparison** : Screenshot de la page de comparaison MLflow → `mlflow_comparison.png`

## Remplir les valeurs dans le rapport

Dans le fichier `.tex`, remplacez les `[VALEUR]` par les résultats réels obtenus après exécution des notebooks :

- `[VALEUR]` → Résultats numériques (accuracy, F1-score, silhouette, etc.)
- `[LISTE]` → Liste des features importantes
- `[OBSERVATIONS]` → Observations/commentaires sur les résultats
- `[MODÈLE]` → Nom du meilleur modèle

## Conseils

1. **Screenshots** : Utilisez des captures d'écran de bonne qualité (PNG recommandé)
2. **Résolution** : Les figures doivent être lisibles (minimum 300 DPI pour impression)
3. **Noms de fichiers** : Respectez exactement les noms de fichiers indiqués dans le `.tex`
4. **Compilation** : Compilez deux fois pour que les références (tableaux, figures) soient correctes

## Résultat final

Après compilation, vous obtiendrez `rapport_wine_quality.pdf` avec toutes les sections, figures et résultats du projet.
