### ---- IMPORTS ---- ###
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture



### ---- RANDOM SEED ---- ###
np.random.seed(42) # para reproducibilidad



### ---- FUNCIONES GRAFICAR ---- ###

## GRAFICAR DATASET TARGET
def graph_init_data(dataset, df_scaled):

    # figura 3d
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b'] # colores 
    class_names = dataset.target_names # nombres clases

    # por cada clase objetivo, grafica
    for target, color, name in zip([0, 1, 2], colors, class_names):
        subset = df_scaled[df_scaled['target'] == target] # filtra el dataset por clase
        ax.scatter(subset['sepal length (scaled)'], 
                subset['petal length (scaled)'], 
                subset['petal width (scaled)'], 
                c=color, 
                label=name,
                s=40)  # s es el tamaño de los puntos

    #  info gráfica
    ax.set_xlabel('Sepal Length (scaled)') # eje x
    ax.set_ylabel('Petal Length (scaled)') # eje y
    ax.set_zlabel('Petal Width (scaled)') # eje z
    ax.set_title('Iris dataset con clases objetivo') # título
    ax.legend() # leyenda con las clases

    plt.tight_layout() # ajusta el layout
    plt.show() # muestra la gráfica


## GRAFICAR CLUSTERING KMEANS
def graph_kmeans_clustering(df_scaled, clustering, n_clusters):
    
    # labels y centroides
    labels = clustering.labels_
    centers = clustering.cluster_centers_
    
    # figura 3d
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # colores
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'][:n_clusters]
    
    # graficar cada cluster
    for cluster, color in zip(range(n_clusters), colors):
        subset = df_scaled[labels == cluster]
        ax.scatter(
            subset['sepal length (scaled)'],
            subset['petal length (scaled)'],
            subset['petal width (scaled)'],
            c=color,
            label=f'Cluster {cluster}',
            s=40,
            alpha=0.7
        )
    
    # graficar centroides
    ax.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2],
        c='black', marker='X', s=200, label='Centroides'
    )
    
    # ejes, título y leyenda
    ax.set_xlabel('Sepal Length (scaled)')
    ax.set_ylabel('Petal Length (scaled)')
    ax.set_zlabel('Petal Width (scaled)')
    ax.set_title(f'Clustering KMeans ({n_clusters} clusters)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


## GRAFICAR CLUSTERING DBSCAN
def graph_dbscan_clustering(df_scaled, dbscan):

    # labels
    labels = dbscan.labels_
    
    # número de clusters (sin ruido)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # figura 3d
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # colores (clusters y ruido)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange'][:n_clusters]
    
    # graficar cada cluster
    unique_labels = set(labels)
    for label, color in zip(unique_labels, colors + ['k']):
        if label == -1:  # Ruido
            subset = df_scaled[labels == label]
            ax.scatter(
                subset['sepal length (scaled)'],
                subset['petal length (scaled)'],
                subset['petal width (scaled)'],
                c='k',
                marker='x',
                label='Ruido',
                s=40,
                alpha=0.5
            )
        else:
            subset = df_scaled[labels == label]
            ax.scatter(
                subset['sepal length (scaled)'],
                subset['petal length (scaled)'],
                subset['petal width (scaled)'],
                c=color,
                label=f'Cluster {label}',
                s=40,
                alpha=0.7
            )
    
    # ejes, título y leyenda
    ax.set_xlabel('Sepal Length (scaled)')
    ax.set_ylabel('Petal Length (scaled)')
    ax.set_zlabel('Petal Width (scaled)')
    ax.set_title(f'Clustering DBSCAN ({n_clusters} clusters + ruido)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


## GRAFICAR CLUSTERING GAUSSIAN MIXTURE
def graph_gauss_mix_clustering(df_scaled, gmm, n_components):

    # labels
    labels = gmm.predict(df_scaled)
    
    # centroides (medias de los gaussianos)
    centers = gmm.means_
    
    # figura 3d
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # colores
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'][:n_components]
    
    # graficar cada cluster
    for cluster, color in zip(range(n_components), colors):
        subset = df_scaled[labels == cluster]
        ax.scatter(
            subset['sepal length (scaled)'],
            subset['petal length (scaled)'],
            subset['petal width (scaled)'],
            c=color,
            label=f'Cluster {cluster}',
            s=40,
            alpha=0.7
        )
    
    # graficar centroides (medias de los gaussianos)
    ax.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2],
        c='black', marker='X', s=200, label='Centros Gaussianos'
    )
    
    # ejes, título y leyenda
    ax.set_xlabel('Sepal Length (scaled)')
    ax.set_ylabel('Petal Length (scaled)')
    ax.set_zlabel('Petal Width (scaled)')
    ax.set_title(f'Clustering GaussianMixture ({n_components} componentes)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()



### ---- MAIN ---- ###
if __name__ == "__main__":

    ## Análisis dataset
    iris = datasets.load_iris() # carga datset
    df = pd.DataFrame(iris.data, columns=iris.feature_names) # dataframe
    df['target'] = iris.target # añade la variable objetivo

    ## Eliminamos sepal_width (variable menos relevante)
    df.drop(columns=['sepal width (cm)'], inplace=True)

    ## Normalizar SOLO las características (excluyendo 'target')
    scaler = MinMaxScaler()
    features = df.columns.drop('target')  # columnas a normalizar
    df_scaled = df.copy()  # copia del DataFrame original
    df_scaled[features] = scaler.fit_transform(df[features])  # normaliza solo features

    ## Graficar dataset con clases objetivo
    graph_init_data(iris, df_scaled)


    # Eliminamos la variable objetivo para hacer clustering
    df_scaled.drop(columns=['target'], inplace=True)


    ## --- KMEANS CLUSTERING --- ##
    kmeans = KMeans(n_clusters=3, init='k-means++') # inicialización con k-means++
    kmeans.fit(df_scaled) # ajusta modelo
    kmeans_labels = kmeans.labels_ # etiquetas del clustering

    graph_kmeans_clustering(df_scaled, kmeans, 3) # graficar clustering KMeans


    ## --- DBSCAN CLUSTERING --- ##
    eps = .1 # dist vecindario
    min_samples = 8 # min muestras cluster

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(df_scaled)
    dbscan_labels = dbscan.labels_ # etiquetas del clustering

    graph_dbscan_clustering(df_scaled, dbscan) # graficar clustering DBSCAN


    ## --- GAUSSIAN MIXTURE CLUSTERING --- ##
    gauss_mixture = GaussianMixture(n_components=3, covariance_type='full')
    gauss_mixture.fit(df_scaled)
    gauss_mix_labels = gauss_mixture.predict(df_scaled)

    graph_gauss_mix_clustering(df_scaled, gauss_mixture, 3) # graficar clustering Gaussian Mixture
    

    ## --- EVALUAR CLUSTERING --- ##

    ## SILHOUETTE SCORE ##
    # KMeans
    silhouette_kmeans = silhouette_score(df_scaled, kmeans_labels)
    print(f"Silhouette Score KMeans: {silhouette_kmeans:.3f}")

    # DBSCAN (sin ruido)
    if len(set(dbscan_labels)) > 1:  # Necesita al menos 2 clusters
        silhouette_dbscan = silhouette_score(df_scaled[dbscan_labels != -1], 
                                        dbscan_labels[dbscan_labels != -1])
        print(f"Silhouette Score DBSCAN: {silhouette_dbscan:.3f}")

    # Gaussian Mixture
    silhouette_gmm = silhouette_score(df_scaled, gauss_mix_labels)
    print(f"Silhouette Score GMM: {silhouette_gmm:.3f}")


    ## CALINSKI-HARABASZ SCORE ##
    # KMeans
    ch_kmeans = calinski_harabasz_score(df_scaled, kmeans_labels)
    print(f"Calinski-Harabasz KMeans: {ch_kmeans:.3f}")

    # DBSCAN (sin ruido)
    if len(set(dbscan_labels)) > 1:
        ch_dbscan = calinski_harabasz_score(df_scaled[dbscan_labels != -1], 
                                        dbscan_labels[dbscan_labels != -1])
        print(f"Calinski-Harabasz DBSCAN: {ch_dbscan:.3f}")

    # Gaussian Mixture
    ch_gmm = calinski_harabasz_score(df_scaled, gauss_mix_labels)
    print(f"Calinski-Harabasz GMM: {ch_gmm:.3f}")


    ## ADJUSTED RAND SCORE ##
    # Labels verdaderos (etiquetas originales)
    true_labels = iris.target

    # KMeans
    ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
    print(f"Adjusted Rand Index KMeans: {ari_kmeans:.3f}")

    # DBSCAN
    ari_dbscan = adjusted_rand_score(true_labels, dbscan_labels)
    print(f"Adjusted Rand Index DBSCAN: {ari_dbscan:.3f}")

    # Gaussian Mixture
    ari_gmm = adjusted_rand_score(true_labels, gauss_mix_labels)
    print(f"Adjusted Rand Index GMM: {ari_gmm:.3f}")