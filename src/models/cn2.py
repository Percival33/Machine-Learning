import Orange
from src.data.create_data import create_data
import datetime


def analyze_results(results: Orange.evaluation.Results):
    print(
        f"train imte:\t{results.train_time}\ntest_time:\t{results.test_time}\naccuracy:\t{Orange.evaluation.CA(results)}\nrecall:\t{Orange.evaluation.Recall(results)}\nF1:\t{Orange.evaluation.F1(results)}"
    )


def run(file_path, test_split=0.8):
    data = Orange.data.Table(file_path)
    new_data = create_data(data)
    train_data, test_data = Orange.evaluation.testing.sample(new_data, n=test_split)
    learners = [Orange.classification.CN2Learner()]
    results = Orange.evaluation.testing.TestOnTestData(train_data, test_data, learners)
    with open(f'../../.logs/{str(datetime.datetime.now()).replace(" ", "_")}', 'w') as f:
        for r in learners[0].rule_list:
            f.write(r, r.curr_class_dist.tolist())

        f.write(analyze_results(results))

    print(analyze_results(results))


if __name__ == "__main__":
    # data = Orange.data.Table("../../data/raw/adult/adult.data")
    # new_data = create_data(data)
    # print(new_data.domain.class_var)
    #
    # train_data, test_data = Orange.evaluation.testing.sample(new_data, n=0.8)
    # learners = [Orange.classification.CN2Learner()]
    # results = Orange.evaluation.testing.TestOnTestData(train_data, test_data, learners)
    # analyze_results(results)

    run("../../data/processed/test.csv")
