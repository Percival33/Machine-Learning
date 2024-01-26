import os.path

import Orange
from src.data.create_data import create_data
import datetime


def analyze_results(results: Orange.evaluation.Results):
    return f"train imte:\t{results.train_time}\ntest_time:\t{results.test_time}\naccuracy:\t{Orange.evaluation.CA(results)}\nrecall:\t{Orange.evaluation.Recall(results)}\nF1:\t{Orange.evaluation.F1(results)}"


class CN2_Runner:
    def __init__(self):
        self.results = None

    def _log_results(self):
        filepath = os.path.abspath(
            os.path.join(
                __file__,
                "..",
                "..",
                "..",
                ".logs",
                f'{str(datetime.datetime.now()).replace(" ", "_")}.txt',
            )
        )
        with open(filepath, "w") as f:
            for r in self.results.models[0][0].rule_list:
                print(str(r))
                f.write(str(r) + "\n")
            f.write(analyze_results(self.results))

    def fit_predict(self, train_path, test_path, base_rules=None):
        train_data = self._preprocess_data(train_path)
        test_data = self._preprocess_data(test_path)
        learners = [Orange.classification.CN2Learner(base_rules=base_rules)]
        self.results = Orange.evaluation.testing.TestOnTestData(
            data=train_data, test_data=test_data, learners=learners, store_models=True
        )

        self._log_results()

    def get_rules(self):
        return self.results.models[0][0].rule_list

    def get_stats(self):
        return f"train imte:\t{self.results.train_time}\ntest_time:\t{self.results.test_time}\naccuracy:\t{Orange.evaluation.CA(self.results)}\nrecall:\t{Orange.evaluation.Recall(self.results)}\nF1:\t{Orange.evaluation.F1(self.results)}"

    @staticmethod
    def _preprocess_data(file_path):
        data = Orange.data.Table(file_path)
        return create_data(data)


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
