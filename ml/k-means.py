import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    full_dataset = pd.read_csv("../../Dataset/trafficDataset.csv")

    sample_dataset = full_dataset.sample(60400)

    data = sample_dataset

    for n_c in [2, 3, 5, 7]:
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

        for percentile in (70, 80, 90, 95):
            # getting outliers whose distances are greater than some percentile
            outliers = points[np.where(distances > np.percentile(distances, percentile))]

            anom_outliers = outliers[np.where(outliers[:, 796] == "Anomalous")]
            normal_outliers = outliers[np.where(outliers[:, 796] == "Normal")]

            print("Percentile ", percentile, " Number of outliers: ", len(outliers), " anoms: ", len(anom_outliers), " normal: ", len(normal_outliers))

    return


if __name__ == "__main__":
    main()
