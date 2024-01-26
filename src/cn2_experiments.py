import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.cn2_runner import CN2_Runner

if __name__ == "__main__":
    cn2 = CN2_Runner()

    # df = pd.read_csv("../data/processed/adult.csv").head(10_000)
    # df['y'] = df['y'].map(lambda x: x == ' <=50K')

    df = pd.read_csv("../data/processed/bank-full.csv").head(10_000)
    df["y"] = df["y"].map(lambda x: x == "yes")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns="y"), df["y"], test_size=0.2, random_state=21
    )
    pd.concat([X_train, y_train], axis=1).to_csv("tmp-train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv("tmp-test.csv", index=False)

    cn2.fit_predict("tmp-train.csv", "tmp-test.csv")

    for r in cn2.get_rules():
        print(r)
    print(cn2.get_stats())
