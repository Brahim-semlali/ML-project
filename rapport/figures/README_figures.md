# Guide pour ajouter les figures dans ce dossier

Placez tous vos screenshots/figures dans ce dossier `figures/` avec les noms exacts suivants :

## Checklist des figures nécessaires

### ✅ EDA (Exploration des données)
- [ ] `eda_histogrammes.png` - Histogrammes des features
- [ ] `eda_correlation.png` - Matrice de corrélation

### ✅ Réduction de dimension
- [ ] `pca_2d.png` - PCA 2 composantes
- [ ] `pca_3d.png` - PCA 3 composantes
- [ ] `tsne_2d.png` - t-SNE 2D
- [ ] `nmf_2d.png` - NMF 2 composantes
- [ ] `lda_2d.png` - LDA 2D (réduction supervisée)

### ✅ Clustering
- [ ] `kmeans_clusters.png` - K-Means clusters
- [ ] `kmeans_comparison.png` - Comparaison K-Means vs qualité
- [ ] `agg_clusters.png` - Agglomerative clusters
- [ ] `agg_comparison.png` - Comparaison clusters vs qualité réelle
- [ ] `dendrogram.png` - Dendrogramme
- [ ] `dbscan_clusters.png` - DBSCAN clusters
- [ ] `dbscan_outliers.png` - DBSCAN outliers
- [ ] `dbscan_comparison.png` - Comparaison DBSCAN vs qualité réelle
- [ ] `gmm_bic.png` - Courbe BIC pour sélection du nombre de composantes GMM
- [ ] `gmm_clusters.png` - GMM clusters

### ✅ Classification
- [ ] `logistic_confusion_matrix.png` - Logistic Regression CM
- [ ] `naivebayes_confusion_matrix.png` - Naive Bayes CM et courbe ROC
- [ ] `knn_performance.png` - KNN courbe performance
- [ ] `knn_confusion_matrix.png` - KNN CM
- [ ] `decisiontree_tree.png` - Structure de l'arbre de décision
- [ ] `decisiontree_feature_importance.png` - Decision Tree importance
- [ ] `decisiontree_confusion_matrix.png` - Decision Tree CM
- [ ] `svm_kernels.png` - SVM test kernels
- [ ] `svm_confusion_matrix.png` - SVM CM
- [ ] `randomforest_hyperparams.png` - Random Forest hyperparams
- [ ] `randomforest_feature_importance_binary.png` - RF importance (binaire)
- [ ] `randomforest_confusion_matrix_binary.png` - RF CM (binaire)
- [ ] `randomforest_feature_importance_multiclass.png` - RF importance (multi-class)
- [ ] `randomforest_confusion_matrix_multiclass.png` - RF CM (multi-class)
- [ ] `randomforest_comparison.png` - RF comparaison binaire vs multi-class
- [ ] `adaboost_feature_importance.png` - AdaBoost importance
- [ ] `adaboost_confusion_matrix.png` - AdaBoost CM
- [ ] `neuralnetwork_curves.png` - MLP : courbe de loss et ROC
- [ ] `neuralnetwork_confusion_matrix.png` - MLP matrice de confusion
- [ ] `gradientboosting_feature_importance.png` - Gradient Boosting importance
- [ ] `gradientboosting_confusion_matrix.png` - Gradient Boosting CM

### ✅ MLflow
- [ ] `mlflow_experiments.png` - Vue d'ensemble MLflow
- [ ] `mlflow_comparison.png` - Comparaison MLflow

## Comment capturer les figures

1. **Depuis Jupyter Notebook** :
   - Exécuter la cellule qui génère la figure
   - Clic droit sur la figure → "Save image as..." → Sauvegarder avec le bon nom

2. **Depuis MLflow UI** :
   - Ouvrir `http://127.0.0.1:5000`
   - Faire une capture d'écran (Windows: Win+Shift+S, Mac: Cmd+Shift+4)
   - Sauvegarder avec le bon nom

3. **Alternative** :
   - Les notebooks sauvegardent déjà certaines figures automatiquement (ex: `pca_2d.png`, `kmeans_clusters.png`, etc.)
   - Copiez ces fichiers depuis le dossier où le notebook s'exécute vers `rapport/figures/`

## Vérification

Une fois toutes les figures ajoutées, compilez le rapport LaTeX et vérifiez que toutes les figures s'affichent correctement.
