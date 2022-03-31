import pandas as pd
from sklearn.decomposition import PCA

def main():

    df = pd.read_csv("../dataset/trafficDataset.csv")

    pca = PCA(n_components=72)

    principalComponents = pca.fit(df)

    return

if __name__ == "__main__":

    main()