import pandas as pd
from sklearn.decomposition import PCA


def get_all_outliers(MSE_score, threshold=2):
    outliers = []

    for k, value in MSE_score.ge(threshold).iteritems():

        if value == True:
            outliers.append(k)

    return outliers


def main():
    df = pd.read_csv("../../Dataset/trafficDataset.csv")

    anomalous_data  = df.loc[df["REQUEST_TYPE"] == "Anomalous"]
    normal_data = df.loc[df["REQUEST_TYPE"] == "Normal"]

    sample_original_data = normal_data.sample(20000).append(anomalous_data.sample(2000))

    sample_data = sample_original_data.drop("REQUEST_TYPE", axis=1)
    sample_data = sample_data.drop("FULL_REQUEST", axis=1)

    for i in range(128, 720):
        pca = PCA(n_components=i)

        pc_dataset_72 = pca.fit_transform(sample_data)

        inverse_transform_dataset_72 = pca.inverse_transform(pc_dataset_72)

        mse_score = ((sample_data - inverse_transform_dataset_72) ** 2).sum(axis=1)

        for threshold in range(1, 13):

            count_normal = 0

            outliers_indexes = get_all_outliers(mse_score, threshold)

            outliers = [(sample_original_data.loc[key]["FULL_REQUEST"], sample_original_data.loc[key]["REQUEST_TYPE"])
                        for key in
                        outliers_indexes]

            for outlier in outliers:
                # print(outlier[0], " - ", outlier[1])
                if outlier[1] == "Normal":
                    count_normal += 1

            print(i, " - Threshold ", threshold, " Normal: ", count_normal, " Anomalous: ",
                  len(outliers) - count_normal)

    return


if __name__ == "__main__":
    main()
