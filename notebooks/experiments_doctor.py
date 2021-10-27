#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn import datasets


#%%
def get_centroids(points: ndarray, clusters: ndarray):
    centroids = []
    for cluster in np.unique(clusters):
        centroids.append(
            np.mean(
                [points[i] for i in range(len(clusters)) if clusters[i] == cluster],
                axis=0,
            )
        )
    return np.array(centroids)


def get_all_distances(points, centroids):
    return np.array(
        [
            [np.linalg.norm(point - centroid, 2) for centroid in centroids]
            for point in points
        ]
    )


def get_decisions(points, centroids, gamma):
    decisions = []
    for point_distances in get_all_distances(points, centroids):
        left_score = np.power(np.exp(-point_distances).sum(), 2)
        right_score = (gamma - 1) * np.exp(-2 * point_distances).sum()
        decisions.append(left_score > right_score)
    return decisions


def plot_decision(points, decisions, centroids):
    outliers = np.array([points[i] for i in range(len(points)) if decisions[i]])
    inliers = np.array([points[i] for i in range(len(points)) if not decisions[i]])

    plt.plot(inliers.transpose()[0], inliers.transpose()[1], "b.")
    plt.plot(outliers.transpose()[0], outliers.transpose()[1], "m.")
    plt.plot(centroids.transpose()[0], centroids.transpose()[1], "r.")
    plt.show()


#%%
support_points, support_clusters = datasets.make_blobs(
    n_samples=100,
    n_features=2,
    cluster_std=1.5,
)


#%%

support_centroids = get_centroids(support_points, support_clusters)

plt.plot(support_points.transpose()[0], support_points.transpose()[1], ".")
plt.plot(support_centroids.transpose()[0], support_centroids.transpose()[1], ".")
plt.show()

#%%
GAMMA = 2.6

#%%
support_decisions = get_decisions(support_points, support_centroids, GAMMA)


plot_decision(support_points, support_decisions, support_centroids)

#%%
query_points = np.random.uniform(-30, 30, (10000, 2))

#%%
query_decisions = get_decisions(query_points, support_centroids, 2.05)

plot_decision(query_points, query_decisions, support_centroids)
