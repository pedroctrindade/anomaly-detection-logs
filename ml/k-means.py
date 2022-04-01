import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score


def main():
    full_dataset = pd.read_csv("../../Dataset/trafficDataset.csv")

    sample_dataset = full_dataset.sample(5000)

    data = sample_dataset

    for n_c in [3, 5, 7]:
        km = KMeans(n_clusters=n_c, random_state=10)
        clusters = km.fit_predict(data.iloc[:, :-2].values)
        silhouette_avg = silhouette_score(data.iloc[:, :-2].values, clusters)

        print(
            "For n_clusters =",
            n_c,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        centroids = km.cluster_centers_

        points = np.empty((0, len(data.iloc[:, ].values[0])), float)

        distances = np.empty((0, len(data.iloc[:, :-2].values[0])), float)

        for i, center_elem in enumerate(centroids):
            distances = np.append(distances,
                                  cdist([center_elem], data.iloc[:, :-2].values[clusters == i], 'euclidean'))
            points = np.append(points, data.iloc[:, ].values[clusters == i], axis=0)


        percentile = 90
        # getting outliers whose distances are greater than some percentile
        outliers = points[np.where(distances > np.percentile(distances, percentile))]

        print("Number of outliers: ", len(outliers))

    return


if __name__ == "__main__":
    main()
