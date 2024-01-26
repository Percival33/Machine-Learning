import Orange
from src.data.create_data import create_data

# train_data, test_data = Orange.evaluation.testing.sample(new_data, n=0.8)

BANK_FULL_DATA = "../../data/raw/bank+marketing/bank/bank-full.csv"
BANK_SAMPLE_DATA = "../../data/raw/bank+marketing/bank/bank.csv"
ADULT_DATA = "../../data/raw/adult/adult.data"


def analyze_results(results: Orange.evaluation.Results):
    print(
        f"train imte:\t{results.train_time}\ntest_time:\t{results.test_time}\naccuracy:\t{Orange.evaluation.CA(results)}\nrecall:\t{Orange.evaluation.Recall(results)}\nF1:\t{Orange.evaluation.F1(results)}"
    )


if __name__ == "__main__":
    data = Orange.data.Table(ADULT_DATA)
    new_data = create_data(data)
    print(new_data.domain.class_var)

    train_data, test_data = Orange.evaluation.testing.sample(new_data, n=0.8)
    learners = [Orange.classification.CN2Learner()]
    results = Orange.evaluation.testing.TestOnTestData(train_data, test_data, learners)
    analyze_results(results)
