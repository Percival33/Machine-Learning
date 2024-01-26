import heapq
import pandas as pd
from copy import deepcopy
from tqdm import tqdm


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
    def _cover_positive(rule: Rule, example):
        return all(not rule.contains(x, example[x]) for x in example.index)

    def covers(self, example):
        return all(self._cover_positive(rule, example) for rule in self.rules)

    def add(self, rules: set[Rule], positives: pd.DataFrame):
        for rule in rules:
            covered = positives.apply(lambda x: self._cover_positive(rule, x), axis=1).sum()
            if covered > self.covered:
                self.covered = covered
            rule.covered = covered
            self.rules.add(rule)

    def prune_worst(self):
        if self.max_rules is None:
            return
        self.rules = set(heapq.nlargest(self.max_rules, self.rules, key=lambda x: x.covered))


class AQClassifier:
    def __init__(self, **kwargs):
        self.max_star_it = kwargs['star_it']
        self.max_it = kwargs['it']
        self.max_cpx = kwargs['max_cpx']
        self.max_rules = kwargs['max_rules']
        self.cover = Cover(self.max_rules)

    def fit(self, x_train, y_train):
        self.cover = Cover(self.max_rules)
        pos = x_train[y_train]
        neg = x_train[~ y_train]
        work_df = pos
        for _ in tqdm(range(self.max_it)):
            seed = work_df.sample(n=1).iloc[0]
            rules = self._generate_stars(seed, neg)
            self.cover.add(rules, pos)
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
                ps.covered = negatives.apply(lambda x: new_ps.cover(x), axis=1).sum()
                candidates.add(ps)

            # Add new rules
            partial_stars = partial_stars.union(candidates)

            if len(partial_stars) > self.max_cpx:
                partial_stars = set(heapq.nlargest(self.max_cpx, partial_stars, key=lambda x: x.covered))
            covered = work_df.apply(lambda x: any(r.cover(x) for r in partial_stars), axis=1)
            work_df = work_df[~covered]
            i += 1
        return partial_stars