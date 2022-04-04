import pandas as pd
from sklearn.decomposition import PCA

THRESHOLD = 2.0


def get_all_outliers(MSE_score):
    outliers = []

    for k, value in MSE_score.ge(THRESHOLD).iteritems():

        if value == True:
            outliers.append(k)

    return outliers


def main():
    df = pd.read_csv("../../Dataset/trafficDataset.csv")

    sample_original_data = df.sample(40000)

    sample_data = sample_original_data.drop("REQUEST_TYPE", axis=1)
    sample_data = sample_data.drop("FULL_REQUEST", axis=1)

    pca = PCA(n_components=72)

    pc_dataset_72 = pca.fit_transform(sample_data)

    inverse_transform_dataset_72 = pca.inverse_transform(pc_dataset_72)

    MSE_score = ((sample_data - inverse_transform_dataset_72) ** 2).sum(axis=1)

    outliers_indexes = get_all_outliers(MSE_score)

    outliers = [(sample_original_data.loc[key]["FULL_REQUEST"], sample_original_data.loc[key]["REQUEST_TYPE"]) for key in
                outliers_indexes]

    count_normal = 0
    for outlier in outliers:
        print(outlier[0], " - ", outlier[1])
        if outlier[1] == "Normal":
            count_normal += 1

    print("Normal: ", count_normal, " Anomalous: ", len(outliers) - count_normal)
    return


if __name__ == "__main__":
    main()
