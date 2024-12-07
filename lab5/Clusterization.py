from lab1.AdditiveModel import AdditiveModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull


class Clusterization:

    @staticmethod
    def get_kmeans_clusters(data, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)

        # Додаємо кластери до даних
        labels = kmeans.labels_

        plt.figure(figsize=(10, 6))
        for cluster in np.unique(labels):
            cluster_points = data[labels == cluster]
            cluster_color = plt.cm.viridis(cluster / n_clusters)

            # Побудова зони з кольором кластера
            if len(cluster_points) > n_clusters:
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]  # Вершини ConvexHull
                plt.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.3, color=cluster_color)  # Заповнений контур

            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Кластер {cluster}")

            # Побудова контуру
            if len(cluster_points) > n_clusters:
                hull = ConvexHull(cluster_points)
                for simplex in hull.simplices:
                    plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], '--', linewidth=1,
                             color=cluster_color)

        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='+',
                    label='Центри кластерів')

        plt.title("Кластеризація (k-means)")
        plt.xlabel("Часова складова")
        plt.ylabel("Значення")
        plt.legend()
        plt.grid(True)
        plt.show()

        return labels

    @staticmethod
    def get_knn(data, n_clusters=3, n_neighbors=3):
        initial_centroids = data[np.random.choice(data.shape[0], n_clusters, replace=False)]

        # Крок 2: Призначення точок до найближчого центру (початкові мітки)
        nbrs = NearestNeighbors(n_neighbors=1).fit(initial_centroids)
        labels = nbrs.kneighbors(data, return_distance=False).flatten()

        for iteration in range(10):  # Лімітуємо кількість ітерацій
            # Крок 3: Обчислення нових центрів кластерів
            new_centroids = np.array([data[labels == cluster].mean(axis=0) for cluster in range(n_clusters)])

            # Крок 4: Призначення точок до найближчого центру
            nbrs = NearestNeighbors(n_neighbors=1).fit(new_centroids)
            new_labels = nbrs.kneighbors(data, return_distance=False).flatten()

            # Якщо мітки більше не змінюються, завершуємо ітерації
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

        # Візуалізація результатів
        plt.figure(figsize=(10, 6))
        for cluster in range(n_clusters):
            cluster_points = data[labels == cluster]
            cluster_color = plt.cm.viridis(cluster / n_clusters)

            # Побудова зони з кольором кластера
            if len(cluster_points) > n_neighbors:
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                plt.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.3, color=cluster_color)

            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Кластер {cluster}")

            # Побудова контуру
            if len(cluster_points) > n_neighbors:
                for simplex in hull.simplices:
                    plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], '--', linewidth=1,
                             color=cluster_color)

        # Відображення центрів кластерів
        plt.scatter(new_centroids[:, 0], new_centroids[:, 1], s=200, c='red', marker='+', label="Центри кластерів")

        plt.title("Кластеризація (KNN)")
        plt.xlabel("Часова складова")
        plt.ylabel("Значення")
        plt.legend()
        plt.grid(True)
        plt.show()

        return labels


if __name__ == '__main__':
    additive_model = AdditiveModel(lowest_error_border=-150,
                                   highest_error_border=200,
                                   size=100,
                                   constant_value=30,
                                   change_rate=0)

    data = additive_model.generate_experimental_data()
    data_with_features = np.column_stack((
        np.arange(len(data)),  # Часова складова
        data  # Значення з Адитивної моделі
    ))
    print("Вхідні дані (ТОП-10):\n", data_with_features[:10])

    Clusterization.get_kmeans_clusters(data_with_features, 4)
    Clusterization.get_knn(data_with_features, 3, 5)
