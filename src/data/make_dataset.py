import pandas as pd

if __name__ == "__main__":
    ADULT_DATA_HEADER = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education - num",
        "marital - status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital - gain",
        "capital - loss",
        "hours - per - week",
        "native - country",
        "y",
    ]
    ADULT_DATA = "../../data/raw/adult/adult.data"
    ADULT_DATA_TEST = "../../data/raw/adult/adult.test"
    BANK_DATA = "../../data/raw/bank+marketing/bank/bank-full.csv"
    BANK_SAMPLE_DATA = "../../data/raw/bank+marketing/bank/bank.csv"
    PROCESSED_FOLDER = "../../data/processed/"

    adult_df = pd.read_csv(ADULT_DATA, names=ADULT_DATA_HEADER)
    adult_df_test = pd.read_csv(ADULT_DATA_TEST, header=1, names=ADULT_DATA_HEADER)
    adult_df.to_csv(PROCESSED_FOLDER + "adult.csv", index=False)
    adult_df_test.to_csv(PROCESSED_FOLDER + "adult_test.csv", index=False)

    bank_df = pd.read_csv(BANK_DATA, sep=";")
    bank_df = bank_df.replace("unknown", None)
    bank_df.to_csv(PROCESSED_FOLDER + "bank-full.csv", index=False)

    bank_sample_df = pd.read_csv(BANK_SAMPLE_DATA, sep=";")
    bank_sample_df = bank_sample_df.replace("unknown", None)
    bank_sample_df.to_csv(PROCESSED_FOLDER + "bank.csv", index=False)
