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
        self.predicate_value = False
        self.covered = 0

    def add(self, col_name: str, value: str, predicate: bool):
        self._complexes[col_name].add(value)
        self.predicate_value = predicate

    # Check if rule covers example
    # Empty set of allowed values means all are possible
    def cover(self, example):
        for col_idx, val in example.items():
            allowed_neg_values = self._complexes[col_idx]
            if len(allowed_neg_values) == 0:
                continue
            if val not in allowed_neg_values:
                return False
        return True

    def contains(self, col_name: str, value):
        if col_name not in self._complexes.keys():
            return False
        return value in self._complexes[col_name]

    def __hash__(self):
        items = tuple(sorted((k, frozenset(v)) for k, v in self._complexes.items()))
        return hash(items)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return f'{str(self._complexes)}->{self.predicate_value}'

    def __repr__(self):
        return str(self)


def generate_partial_stars(diff_columns, all_columns, predicate: bool) -> set[Rule]:
    partial_stars = set()
    for col_idx, val in diff_columns.items():
        r = Rule(all_columns)
        r.add(col_idx, val, predicate)
        r.covered = 1
        partial_stars.add(r)
    return partial_stars


class Cover:
    def __init__(self, max_rules=10):
        self.rules: set[Rule] = set()
        self.max_rules = max_rules
        self.covered = 0

    @staticmethod
    def cover_positive(rule: Rule, example):
        return all(not rule.contains(x, example[x]) for x in example.index)

    def covers(self, example):
        return any(rule.cover(example) for rule in self.rules)

    def predict(self, example):
        for rule in self.rules:
            if rule.cover(example):
                return rule.predicate_value
        return False

    def add(self, rules: set[Rule], positives: pd.DataFrame):
        for rule in rules:
            covered = positives.apply(lambda x: rule.cover(x), axis=1).sum()
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


def seed_star(clf, seed, neg, predicate):
    rules = clf._generate_stars(seed, neg, predicate)
    return rules


class AQClassifier:
    def __init__(self, **kwargs):
        self.max_star_it = kwargs['star_it']
        self.max_it = kwargs['it']
        self.max_cpx = kwargs['max_cpx']
        self.max_rules = kwargs['max_rules']
        self.parrarel_seeds = kwargs['parrarel_seeds']
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
        work_df = x_train
        covered_count = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for _ in tqdm(range(self.max_it)):
                jobs = []
                n = self.parrarel_seeds if len(work_df) > self.parrarel_seeds else len(work_df)
                for idx, seed in work_df.sample(n=n).iterrows():
                    if y_train[idx]:
                        jobs.append(executor.submit(seed_star, self, seed, neg, False))
                    else:
                        jobs.append(executor.submit(seed_star, self, seed, pos, True))
                for job in jobs:
                    rules = job.result()
                    self.cover.add_rules(rules)
                self.cover.prune_worst()
                covered = work_df.apply(lambda x: self.cover.covers(x) & self.cover.predict(x) == y_train[x.name], axis=1)
                covered_count.append(covered.sum())
                work_df = work_df[~covered]
                if len(work_df) == 0:
                    return covered_count
        return covered_count

    def predict(self, x_test):
        return x_test.apply(lambda x: self.cover.predict(x), axis=1)

    def _rand_not_covered(self, rules: set[Rule], df):
        for i in range(10):
            idx = df.sample(n=1).index[0]
            neg_ex = df.loc[idx]
            if not all(rule.cover(neg_ex) for rule in rules):
                return neg_ex
        return df.loc[df.sample(n=1).index[0]]

    def _generate_stars(self, seed, negatives, predicate) -> set[Rule]:
        def examples_covered_rate(star: Rule):
            return negatives.apply(lambda x: star.cover(x), axis=1).mean()

        partial_stars: set[Rule] = set()
        i = 0
        work_df = negatives
        while len(work_df) > 0 and i < self.max_star_it:
            neg_ex = self._rand_not_covered(partial_stars, work_df)
            # Get (column, value) that rule cannot have to not cover seed
            diff_columns = neg_ex[seed != neg_ex]

            if len(partial_stars) == 0:
                for ps in generate_partial_stars(diff_columns, neg_ex.index, predicate):
                    ps.covered = examples_covered_rate(ps)
                    partial_stars.add(ps)
                continue

            # Modify present stars of new values for column
            # Potentially make rule more general
            candidates = set()
            for ps in partial_stars:
                for col_idx, val in diff_columns.items():
                    if not ps.contains(col_idx, val):
                        new_ps = deepcopy(ps)
                        new_ps.add(col_idx, val, predicate)
                        new_ps.covered = examples_covered_rate(new_ps)
                        candidates.add(new_ps)
            # Add rules from current difference between seed and negative example
            for ps in generate_partial_stars(diff_columns, neg_ex.index, predicate):
                ps.covered = examples_covered_rate(ps)
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
    df = pd.read_csv("../../data/test/adult-0.csv")
    df['y'] = df['y'].map(lambda x: x == 0)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='y'), df['y'], test_size=0.2)
    clf = AQClassifier(**{
        'star_it': 50,
        'it': 5,
        'max_cpx': 20,
        'max_rules': 10,
        'parrarel_seeds': 3
    })
    res = clf.fit(X_train, y_train)
    print(res)
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {classification_report(y_test, y_pred)}")
