import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def main():

    dataset = pd.read_csv("../../Dataset/trafficDataset.csv")

    #scaler = MinMaxScaler(feature_range=(0,1))
    #X = scaler.fit_transform(dataset.iloc[:, :-1].values)

    std_scale = StandardScaler()

    X = std_scale.fit_transform(dataset.iloc[:, :-1].values)
    y = dataset.iloc[:, 796].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)



    y_pred = knn_model.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred, average=None))
    print("Recall:", metrics.recall_score(y_test, y_pred, average=None))


    return


if __name__ == "__main__":
    main()