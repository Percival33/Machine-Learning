from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

import pandas as pd


def create_test_data(name: str):
    df = pd.read_csv(f"../../data/processed/{name}.csv")
    lb = LabelEncoder()
    for i in df.columns:
        df[i] = lb.fit_transform(df[i])

    # clusters = KMeans(n_clusters=10, random_state=0, n_init="auto").fit_predict(df)
    CLUSTERS = 20
    clusters = KMeans(n_clusters=CLUSTERS, random_state=0, n_init="auto").fit_predict(df.drop('y', axis=1))

    clustered_data = []
    for i in range(CLUSTERS):
        cluster_df = df[clusters == i]
        all_elem = cluster_df['y'].value_counts()
        clustered_data.append({
            'df': cluster_df,
            'ones': all_elem[1] / all_elem.sum()
        })

    sorted(clustered_data, key=lambda x: x['ones'])

    for i in range(CLUSTERS):
        pd.DataFrame(clustered_data[i]['df']).to_csv(f"../../data/test/{name}-{i}.csv", index=False)


if __name__ == "__main__":
    create_test_data('bank')
    create_test_data('adult')
    print("making test dataset has finished")
