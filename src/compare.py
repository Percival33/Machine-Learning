from models.cn2 import run
import os

if __name__ == "__main__":
    file = os.path.abspath(os.path.join(__file__, "..", "..", "data", "processed", "test.csv"))

    print(run(file))
