import pandas as pd
from sklearn.ensemble import IsolationForest


def main():
    return


if __name__ == "__main__":
    df = pd.read_csv("../../Dataset/trafficDataset.csv")

    anomalous_data = df.loc[df["REQUEST_TYPE"] == "Anomalous"]
    normal_data = df.loc[df["REQUEST_TYPE"] == "Normal"]
    all_data = normal_data

    all_data = all_data.sample(34000)
    all_data = all_data.append(anomalous_data.sample(3400))
    all_data = all_data.sample(37400)
    outliers_counter = len(all_data[all_data['REQUEST_TYPE'] == "Anomalous"])

    model = IsolationForest(contamination=float(0.13),
                            random_state=42)
    model.fit(all_data.iloc[:, :-2].values)

    all_data["scores"] = model.decision_function(all_data.iloc[:, :-2].values)
    all_data['anomaly'] = model.predict(all_data.iloc[:, :-3].values)

    correct_outliers = 0
    wrong_outliers = 0
    for index, row in all_data.iterrows():
        if row["anomaly"] == -1 and row["REQUEST_TYPE"] == "Anomalous":
            correct_outliers += 1
        elif row["anomaly"] != -1 and row["REQUEST_TYPE"] == "Anomalous":
            wrong_outliers += 1

    print("Total outliers: ", outliers_counter)
    print("Wrong outliers: ", wrong_outliers)
    print("Correct outliers: ", correct_outliers)
    print("Recall:", 100 * correct_outliers / outliers_counter)

    main()
