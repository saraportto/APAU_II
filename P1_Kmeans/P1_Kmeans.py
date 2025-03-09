### IMPORTS ###
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

### FUNCTIONS ###
np.random.seed(42)

### CLASE KMEANS from scratch ###
class KMeansScratch:

    ## Atributos
    def __init__(self, n_clusters=4, max_iter=100, init_method='random'):
        '''
        n_clusters: Número de clusters
        max_iter: Número máximo de iteraciones
        init_method: Método de inicialización de los centroides (random, dataset)
        centroids: Coordenadas de los centroides
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init_method = init_method
        self.centroids = None

    ## Obtener los centroides
    def getCentroids(self):
        return ("Centroides: \n" + str(self.centroids))

    ## Inicialización de los centroides
    def initialize_centroids(self, X):
        '''
        X: Datos
        '''

        # Centroides inicializados aleatoiramente dentro de los límites de los datos
        if self.init_method == 'random':
            x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
            y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
            self.centroids = np.column_stack((
                np.random.uniform(x_min, x_max, self.n_clusters),
                np.random.uniform(y_min, y_max, self.n_clusters)
            ))

        # Centroides inicializados con instancias aleatorias del dataset
        elif self.init_method == 'dataset':
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X[indices]

        # Inicialización no válida
        else:
            raise ValueError("Método de inicialización no válido")

    ## Cálculo de la distancia euclidiana entre dos puntos
    def euclidean_distance(self, a, b) -> np.ndarray:
        '''
        a, b: Puntos
        '''
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    ## Asignación de instancias a los clusters
    def assign_clusters(self, X) -> np.ndarray:
        '''
        X: Datos
        '''

        # Distancia de cada instancia a los centroides
        distances = np.array([self.euclidean_distance(X, centroid) for centroid in self.centroids])
        labels = np.argmin(distances, axis=0) # Cluster más cercano a cada instancia

        # EQUILIBRIO de tamaño en caso de EMPATE
        cluster_sizes = np.bincount(labels, minlength=self.n_clusters)
        for i in range(X.shape[0]):
            min_dist_clusters = np.where(distances[:, i] == np.min(distances[:, i]))[0]
            if len(min_dist_clusters) > 1:
                smallest_cluster = min(min_dist_clusters, key=lambda c: cluster_sizes[c])
                labels[i] = smallest_cluster
                cluster_sizes[smallest_cluster] += 1
                cluster_sizes[min_dist_clusters[min_dist_clusters != smallest_cluster]] -= 1

        return labels

    ## Actualización de los centroides
    def update_centroids(self, X, labels) -> np.ndarray:
        '''
        X: Datos
        labels: Clusters asignados a las instancias
        '''
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        for i in range(self.n_clusters):
            if np.any(labels == i):
                new_centroids[i] = X[labels == i].mean(axis=0)
            else:
                new_centroids[i] = self.centroids[i]

        return new_centroids

    ## Entrenamiento del modelo
    def fit(self, X):
        '''
        X: Datos
        '''

        self.initialize_centroids(X)  # inicialización centroides

        # Bucle de entrenamiento (teniendo en cuenta max_iter)
        for _ in range(self.max_iter):
            labels = self.assign_clusters(X)  # asignación de instancias a clusters
            new_centroids = self.update_centroids(X, labels) # actualización de centroides

            # Centroides no cambian --> DETIENE el entrenamiento
            if np.allclose(self.centroids, new_centroids):
                break

            # Actualización de centroides
            self.centroids = new_centroids

        return labels
    

### FUNCIÓN ENCONTRAR NÚMERO ÓPTIMO DE CLUSTERS CON COEFICIENTE DE SILHOUETTE ###
def find_optimal_clusters_silhouette(X, max_k=10):
    '''
    X: Datos
    max_k: Número máximo de clusters
    '''
    # Lista para almacenar Coeficientes de Silhouette
    silhouette_scores = []
    k_values = range(2, max_k)

    # Evaluar distintos k
    for k in k_values:
        kmeans = KMeansScratch(n_clusters=k, max_iter=100, init_method='random')
        labels = kmeans.fit(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    # Obtener el k con el mejor Silhouette Score
    optimal_k = k_values[np.argmax(silhouette_scores)]

    # Graficar datos
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'Óptimo k={optimal_k}')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Silhouette')
    plt.show()

    return optimal_k


### FUNCIÓN ENCONTRAR NÚMERO ÓPTIMO DE CLUSTERS CON ELBOW METHOD ###
### FUNCIÓN PARA ENCONTRAR K ÓPTIMO - ELBOW METHOD ###
def find_optimal_clusters_elbow(X, max_k=10):
    '''
    X: Datos
    max_k: Número máximo de clusters
    '''
    # Lista para almacenar la SSE (inertia)
    sse = []
    k_values = range(2, max_k)

    # Evaluar distintos valores de k
    for k in k_values:
        kmeans = KMeansScratch(n_clusters=k, max_iter=100, init_method='random')
        labels = kmeans.fit(X)

        # Calcular SSE (suma distancias cuadráticas de los puntos a sus centroides)
        inertia = 0
        for i, centroid in enumerate(kmeans.centroids):
            inertia += np.sum((X[labels == i] - centroid) ** 2)

        sse.append(inertia)

    # Encontrar codo
    kneedle = KneeLocator(k_values, sse, curve="convex", direction="decreasing")
    optimal_k = kneedle.elbow

    # Graficar método del codo
    plt.plot(k_values, sse, 'o-')
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("SSE")
    plt.title("Método del Codo para determinar k óptimo")
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'Óptimo k={optimal_k}')
    plt.legend()
    plt.show()

    return optimal_k


### BUCLE PRINCIPAL ###
if __name__ == '__main__':
    ## DATASET
    # Creación dataset
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # Nº instancias
    print(X.shape)

    # Visualizar datos
    plt.scatter(X[:, 0], X[:, 1], s=10)
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.title("Dataset sintético")
    plt.show()

    ## ENCONTRAR K ÓPTIMO - SILHOUETTE SCORE
    optimal_k_silhouette = find_optimal_clusters_silhouette(X, max_k=10)
    print(f"K óptimo con Silhouette Score: {optimal_k_silhouette}")

    ## ENCONTRAR K ÓPTIMO - ELBOW METHOD
    optimal_k_elbow = find_optimal_clusters_elbow(X, max_k=10)
    print(f"K óptimo con Elbow Method: {optimal_k_elbow}")

    ## KMEANS CON INICIALIZACIÓN DE CENTROIDES ALEATORIA Y K ÓPTIMO CON SILHOUETTE
    # Modelo
    kmeans = KMeansScratch(n_clusters=optimal_k_silhouette, max_iter=100, init_method='random')
    labels = kmeans.fit(X)
    print(kmeans.getCentroids())

    # Visualización de los clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=200)
    plt.title("K-Means from Scratch")
    plt.show()

    ## KMEANS CON INICIALIZACIÓN DE CENTROIDES CON INSTANCIAS DEL DATASET Y K ÓPTIMO CON ELBOW METHOD
    # Modelo
    kmeans = KMeansScratch(n_clusters=optimal_k_elbow, max_iter=100, init_method='dataset')
    labels = kmeans.fit(X)
    print(kmeans.getCentroids())

    # Visualización de los clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=200)
    plt.title("K-Means from Scratch")
    plt.show()