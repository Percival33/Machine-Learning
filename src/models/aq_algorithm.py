import heapq
import pickle

import pandas as pd
from copy import deepcopy

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import concurrent.futures


class Rule:
    def __init__(self, col_names: list[str]):
        self._complexes = {name: set() for name in col_names}
        self.covered = 0

    def add(self, col_name: str, value: str):
        self._complexes[col_name].add(value)

    # Check if rule covers negative example
    # Empty set of allowed values means all are possible
    def cover(self, example):
        for col_idx, val in example.items():
            allowed_neg_values = self._complexes[col_idx]
            if len(allowed_neg_values) == 0:
                continue
            if val not in allowed_neg_values:
                return False
        return True

    def update(self, example):
        if self.cover(example):
            self.covered += 1

    def contains(self, col_name: str, value):
        if col_name not in self._complexes.keys():
            return False
        return value in self._complexes[col_name]

    def __hash__(self):
        items = tuple(sorted((k, frozenset(v)) for k, v in self._complexes.items()))
        return hash(items)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __lt__(self, other):
        # Invert the logic for min-heap to work as max-heap
        return self.covered > other.covered

    def __str__(self):
        return str(self._complexes)

    def __repr__(self):
        return str(self)


def generate_partial_stars(diff_columns, all_columns) -> set[Rule]:
    partial_stars = set()
    for col_idx, val in diff_columns.items():
        r = Rule(all_columns)
        r.add(col_idx, val)
        r.covered = 1
        partial_stars.add(r)
    return partial_stars


def update_stars(stars: set[Rule], negative):
    for star in stars:
        if star.cover(negative):
            star.update(negative)


class Cover:
    def __init__(self, max_rules=10):
        self.rules: set[Rule] = set()
        self.max_rules = max_rules
        self.covered = 0

    @staticmethod
    def cover_positive(rule: Rule, example):
        return all(not rule.contains(x, example[x]) for x in example.index)

    def covers(self, example):
        return all(self.cover_positive(rule, example) for rule in self.rules)

    def add(self, rules: set[Rule], positives: pd.DataFrame):
        for rule in rules:
            covered = positives.apply(lambda x: self.cover_positive(rule, x), axis=1).sum()
            if covered > self.covered:
                self.covered = covered
            rule.covered = covered
            self.rules.add(rule)

    def add_rules(self, rules: set[Rule]):
        for rule in rules:
            if rule.covered > self.covered:
                self.covered = rule.covered
            self.rules.add(rule)

    def prune_worst(self):
        if self.max_rules is None:
            return
        self.rules = set(heapq.nlargest(self.max_rules, self.rules, key=lambda x: x.covered))


def seed_star(self, seed):
    rules = self._generate_stars(seed, self.neg)
    for rule in rules:
        rule.covered = self.pos.apply(lambda x: Cover.cover_positive(rule, x), axis=1).sum()
    return rules


class AQClassifier:
    def __init__(self, **kwargs):
        self.max_star_it = kwargs['star_it']
        self.max_it = kwargs['it']
        self.max_cpx = kwargs['max_cpx']
        self.max_rules = kwargs['max_rules']
        self.cover = Cover(self.max_rules)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.cover.rules, file)

    @staticmethod
    def load(filename, **kwargs):
        with open(filename, 'rb') as file:
            rules = pickle.load(file)
            clf = AQClassifier(**kwargs)
            cover = Cover(kwargs['max_rules'])
            cover.rules = rules
            clf.cover = cover
            return clf

    def fit(self, x_train, y_train):
        self.cover = Cover(self.max_rules)
        pos = x_train[y_train]
        neg = x_train[~ y_train]
        self.pos = pos
        self.neg = neg
        work_df = pos

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for _ in tqdm(range(self.max_it)):
                n = 2 if len(work_df) > 2 else len(work_df)
                jobs = [executor.submit(seed_star, self, seed) for _, seed in work_df.sample(n=n).iterrows()]
                for job in jobs:
                    rules = job.result()
                    self.cover.add_rules(rules)
                self.cover.prune_worst()
                covered = work_df.apply(lambda x: self.cover.covers(x), axis=1)
                work_df = work_df[~covered]
                if len(work_df) == 0:
                    return

    def predict(self, x_test):
        return x_test.apply(lambda x: self.cover.covers(x), axis=1)

    def _rand_not_covered(self, rules: set[Rule], df):
        for i in range(10):
            idx = df.sample(n=1).index[0]
            neg_ex = df.loc[idx]
            if any(rule.cover(neg_ex) for rule in rules):
                return neg_ex
        return df.loc[df.sample(n=1).index[0]]

    def _generate_stars(self, seed, negatives) -> set[Rule]:
        partial_stars: set[Rule] = set()
        i = 0
        work_df = negatives
        while len(work_df) > 0 and i < self.max_star_it:
            neg_ex = self._rand_not_covered(partial_stars, work_df)
            # Get (column, value) that rule cannot have to not cover seed
            diff_columns = neg_ex[seed != neg_ex]

            if len(partial_stars) == 0:
                for ps in generate_partial_stars(diff_columns, neg_ex.index):
                    ps.covered = negatives.apply(lambda x: ps.cover(x), axis=1).sum()
                    partial_stars.add(ps)
                continue

            # Modify present stars of new values for column
            # Potentially make rule more general
            candidates = set()
            for ps in partial_stars:
                for col_idx, val in diff_columns.items():
                    if not ps.contains(col_idx, val):
                        new_ps = deepcopy(ps)
                        new_ps.add(col_idx, val)
                        new_ps.covered = negatives.apply(lambda x: new_ps.cover(x), axis=1).sum()
                        candidates.add(new_ps)
            # Add rules from current difference between seed and negative example
            for ps in generate_partial_stars(diff_columns, neg_ex.index):
                ps.covered = negatives.apply(lambda x: ps.cover(x), axis=1).sum()
                candidates.add(ps)

            # Add new rules
            partial_stars = partial_stars.union(candidates)

            if len(partial_stars) > self.max_cpx:
                partial_stars = set(heapq.nlargest(self.max_cpx, partial_stars, key=lambda x: x.covered))
            covered = work_df.apply(lambda x: any(r.cover(x) for r in partial_stars), axis=1)
            work_df = work_df[~covered]
            i += 1
        return partial_stars


if __name__ == '__main__':
    df = pd.read_csv("../../data/processed/adult.csv").select_dtypes(exclude=["number"]).head(100)
    df['y'] = df['y'].map(lambda x: x == ' <=50K')
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='y'), df['y'], test_size=0.2)
    clf = AQClassifier(**{
        'star_it': 20,
        'it': 10,
        'max_cpx': 50,
        'max_rules': 10,
    })
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {classification_report(y_test, y_pred)}")
    clf.save("../../models/aq")
    clf2 = AQClassifier.load("../../models/aq", **{
        'star_it': 20,
        'it': 10,
        'max_cpx': 50,
        'max_rules': 10,
    })
    y_pred2 = clf2.predict(X_test)
    print(f"Accuracy: {classification_report(y_test, y_pred2)}")
