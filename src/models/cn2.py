import Orange
from src.data.create_data import create_data
import datetime


def analyze_results(results: Orange.evaluation.Results):
    return f"train imte:\t{results.train_time}\ntest_time:\t{results.test_time}\naccuracy:\t{Orange.evaluation.CA(results)}\nrecall:\t{Orange.evaluation.Recall(results)}\nF1:\t{Orange.evaluation.F1(results)}"


def run(file_path, test_split=0.8):
    data = Orange.data.Table(file_path)
    new_data = create_data(data)
    train_data, test_data = Orange.evaluation.testing.sample(new_data, n=test_split)
    learners = [Orange.classification.CN2Learner()]
    results = Orange.evaluation.testing.TestOnTestData(
        train_data, test_data, learners, store_models=True
    )
    with open(f'../.logs/{str(datetime.datetime.now()).replace(" ", "_")}.txt', "w") as f:
        for r in results.models[0][0].rule_list:
            print(str(r))
            f.write(str(r) + "\n")
        f.write(analyze_results(results))

    print(analyze_results(results))


if __name__ == "__main__":
    run("../../data/processed/test.csv")
