import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    dataset = pd.read_csv("../../Dataset/trafficDataset.csv")

    anomalous_data = dataset.loc[dataset["REQUEST_TYPE"] == "Anomalous"]
    normal_data = dataset.loc[dataset["REQUEST_TYPE"] == "Normal"]

    dataset = anomalous_data.sample(3000).append(normal_data.sample(5000))

    X = dataset.iloc[:, :-2].values
    y = dataset.iloc[:, 796].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average=None))
    print("Recall:", metrics.recall_score(y_test, y_pred, average=None))

    return


if __name__ == "__main__":
    main()
