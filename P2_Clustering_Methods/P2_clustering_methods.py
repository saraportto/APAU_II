### ---- IMPORTS ---- ###
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN


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

    # Por cada clase objetivo, grafica
    for target, color, name in zip([0, 1, 2], colors, class_names):
        subset = df_scaled[df_scaled['target'] == target] # filtra el dataset por clase
        ax.scatter(subset['sepal length (cm)'], 
                subset['petal length (cm)'], 
                subset['petal width (cm)'], 
                c=color, 
                label=name,
                s=40)  # s es el tamaño de los puntos

    # Info en la gráfica
    ax.set_xlabel('Sepal Length (scaled)') # eje x
    ax.set_ylabel('Petal Length (scaled)') # eje y
    ax.set_zlabel('Petal Width (scaled)') # eje z
    ax.set_title('Iris dataset con clases objetivo') # título
    ax.legend() # leyenda con las clases

    plt.tight_layout() # ajusta el layout
    plt.show() # muestra la gráfica

## GRAFICAR CLUSTERING
def graph_clustering(df_scaled, clustering, n_clusters, clustering_method):
    
    # Labels y centroides
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
            subset['sepal length (cm)'],
            subset['petal length (cm)'],
            subset['petal width (cm)'],
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
    
    # Configuración del gráfico
    ax.set_xlabel('Sepal Length (scaled)')
    ax.set_ylabel('Petal Length (scaled)')
    ax.set_zlabel('Petal Width (scaled)')
    ax.set_title(f'Clustering {clustering_method} ({n_clusters} clusters)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def graph_dbscan_clustering(df_scaled, dbscan):
    # Obtener labels
    labels = dbscan.labels_
    
    # Número de clusters (excluyendo el ruido)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Crear figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colores para los clusters + color especial para ruido
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange'][:n_clusters]
    
    # Graficar cada cluster y el ruido
    unique_labels = set(labels)
    for label, color in zip(unique_labels, colors + ['k']):
        if label == -1:  # Ruido
            subset = df_scaled[labels == label]
            ax.scatter(
                subset['sepal length (cm)'],
                subset['petal length (cm)'],
                subset['petal width (cm)'],
                c='k',
                marker='x',
                label='Ruido',
                s=40,
                alpha=0.5
            )
        else:
            subset = df_scaled[labels == label]
            ax.scatter(
                subset['sepal length (cm)'],
                subset['petal length (cm)'],
                subset['petal width (cm)'],
                c=color,
                label=f'Cluster {label}',
                s=40,
                alpha=0.7
            )
    
    # Configuración del gráfico
    ax.set_xlabel('Sepal Length (scaled)')
    ax.set_ylabel('Petal Length (scaled)')
    ax.set_zlabel('Petal Width (scaled)')
    ax.set_title(f'Clustering DBSCAN ({n_clusters} clusters + ruido)')
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
    graph_clustering(df_scaled, kmeans, 3, 'KMeans') # graficar clustering KMeans


    ## --- DBSCAN CLUSTERING --- ##
    eps = .1
    min_samples = 8

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(df_scaled)
    graph_dbscan_clustering(df_scaled, dbscan) # graficar clustering DBSCAN


    ## --- GAUSSIAN MIXTURE CLUSTERING --- ##
    pass
    