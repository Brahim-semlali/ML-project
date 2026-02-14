"""
Copie les figures des dossiers algorithmes (clustering/KMeans/, etc.) vers rapport/figures/
pour que le rapport LaTeX affiche les images à jour.
Exécuter depuis la racine du projet : python scripts/collect_figures.py
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAPPORT_FIGURES = os.path.join(ROOT, 'rapport', 'figures')

# Mapping: (dossier algorithme, liste de (fichier_local, nom_dans_rapport))
FIGURES_MAP = [
    # Clustering
    (os.path.join(ROOT, 'clustering', 'KMeans'), ['kmeans_clusters.png', 'kmeans_comparison.png']),
    (os.path.join(ROOT, 'clustering', 'AgglomerativeClustering'), ['agg_clusters.png', 'agg_comparison.png', 'dendrogram.png']),
    (os.path.join(ROOT, 'clustering', 'DBSCAN'), ['dbscan_clusters.png', 'dbscan_comparison.png', 'dbscan_outliers.png']),
    (os.path.join(ROOT, 'clustering', 'GMM'), ['gmm_bic.png', 'gmm_clusters.png']),
    # Réduction
    (os.path.join(ROOT, 'reduction', 'PCA'), ['pca_2d.png', 'pca_3d.png']),
    (os.path.join(ROOT, 'reduction', 'tSNE'), ['tsne_2d.png']),
    (os.path.join(ROOT, 'reduction', 'NMF'), ['nmf_2d.png']),
    (os.path.join(ROOT, 'reduction', 'LDA'), ['lda_2d.png']),
    # Classification
    (os.path.join(ROOT, 'classification', 'LogisticRegression'), ['logistic_confusion_matrix.png']),
    (os.path.join(ROOT, 'classification', 'KNN'), ['knn_performance.png', 'knn_confusion_matrix.png']),
    (os.path.join(ROOT, 'classification', 'DecisionTree'), ['decisiontree_feature_importance.png', 'decisiontree_tree.png', 'decisiontree_confusion_matrix.png']),
    (os.path.join(ROOT, 'classification', 'SVM'), ['svm_kernels.png', 'svm_confusion_matrix.png']),
    (os.path.join(ROOT, 'classification', 'RandomForest'), [
        'randomforest_hyperparams.png', 'randomforest_feature_importance_binary.png', 'randomforest_feature_importance_multiclass.png',
        'randomforest_confusion_matrix_binary.png', 'randomforest_confusion_matrix_multiclass.png', 'randomforest_comparison.png'
    ]),
    (os.path.join(ROOT, 'classification', 'GradientBoosting'), ['gradientboosting_feature_importance.png', 'gradientboosting_confusion_matrix.png']),
    (os.path.join(ROOT, 'classification', 'AdaBoost'), ['adaboost_feature_importance.png', 'adaboost_confusion_matrix.png']),
    (os.path.join(ROOT, 'classification', 'NaiveBayes'), ['naivebayes_confusion_matrix.png']),
    (os.path.join(ROOT, 'classification', 'NeuralNetwork'), ['neuralnetwork_curves.png', 'neuralnetwork_confusion_matrix.png']),
]

os.makedirs(RAPPORT_FIGURES, exist_ok=True)
count = 0
for folder, filenames in FIGURES_MAP:
    if not os.path.isdir(folder):
        continue
    for f in filenames:
        src = os.path.join(folder, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(RAPPORT_FIGURES, f))
            count += 1
            print(f"  {f} <- {os.path.basename(folder)}/")
print(f"\n{count} figure(s) copiée(s) vers rapport/figures/")
