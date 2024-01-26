from models.cn2_runner import CN2_Runner
import os

if __name__ == "__main__":
    file = os.path.abspath(os.path.join(__file__, "..", "..", "data", "processed", "test.csv"))
    folder_path = os.path.abspath("../data/test")
    print(folder_path)
    cn2 = CN2_Runner()
    adult_test = list(filter(lambda x: x.startswith("adult"), os.listdir("../data/test")))
    adult_test = sorted(adult_test, key=lambda x: int(x.split('-')[1].split('.')[0]))

    bank_test = list(filter(lambda x: x.startswith("bank"), os.listdir("../data/test")))
    bank_test = sorted(bank_test, key=lambda x: int(x.split('-')[1].split('.')[0]))
    print(adult_test)
    print(bank_test)

    previous_rules = None

    for idx, test_file in enumerate(adult_test):
        test_path = os.path.join(folder_path, test_file)
        prev_path = os.path.join(folder_path, adult_test[max(idx-1, 0)])
        cn2.fit_predict(prev_path, test_path, base_rules=previous_rules)
        print(idx, cn2.get_stats())
        previous_rules = cn2.get_rules()
