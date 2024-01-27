import heapq
import pickle
import time
import json
import pandas as pd
from copy import deepcopy

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import concurrent.futures
from numpy.random import choice


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

    def covers(self, example):
        return any(rule.cover(example) for rule in self.rules)

    def predict(self, example):
        return sum(
            int(rule.predicate_value) if rule.cover(example) else int( not rule.predicate_value)
            for rule in self.rules)/len(self.rules) >= 0.5

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


class AQClassifier:
    def __init__(self, **kwargs):
        self.max_star_it = kwargs['star_it']
        self.max_it = kwargs['it']
        self.max_cpx = kwargs['max_cpx']
        self.max_rules = kwargs['max_rules']
        self.no_diff_it = kwargs['no_diff_it']
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
        pos = x_train[y_train]
        neg = x_train[~ y_train]
        work_idx = x_train.index

        bar = tqdm(range(self.max_it))
        stats = {
            'new_covered': [],
            'time': [],
            'cls_distrib': []
        }

        for _ in bar:
            _start = time.time()
            
            # Choose seed as element of less common class
            less_common_val = y_train[work_idx].value_counts().idxmin()
            idx = choice(y_train[y_train.index.isin(work_idx)][y_train==less_common_val].index, size=1, replace=False)[0]
            
            # For given seed and examples of opposite class generate rules
            seed = x_train.loc[idx]
            predicate = not y_train.loc[idx]
            data = pos[pos.index.isin(work_idx)] if predicate else neg[neg.index.isin(work_idx)]
            rules = self._generate_stars(seed, data, predicate)
            
            # Evaluate newly created rules on rest of df
            for rule in rules:
                rule.covered = x_train.loc[work_idx].apply(lambda x: rule.predicate_value==y_train[x.name] if rule.cover(x) else False, axis=1).mean()
            
            # Add new rules and prune them by their score
            self.cover.add_rules(rules)
            self.cover.prune_worst()
            
            # Remove from current set examples that are predictet correctly
            covered = x_train.loc[work_idx].apply(lambda x: self.cover.predict(x) == y_train[x.name], axis=1)
            work_idx = work_idx[~covered]

            _end = time.time()
            stats['new_covered'].append(int(covered.sum()))
            stats['time'].append(_end - _start)
            stats['cls_distrib'].append(float(y_train.loc[work_idx].mean()))
            
            if len(work_idx) == 0:
                return stats
            # return if result doesn't change for n iterations
            if len(stats['new_covered']) > self.no_diff_it:
                if len(set(stats['new_covered'][-self.no_diff_it:])) <= 1:
                    return stats
                
            bar.set_postfix({'left': len(work_idx), 'class_distrb': stats['cls_distrib'][-1]})
        return stats

    def predict(self, x_test):
        return x_test.apply(lambda x: self.cover.predict(x), axis=1)

    def _rand_not_covered(self, rules: set[Rule], df):
        for _ in range(10):
            idx = df.sample(n=1).index[0]
            neg_ex = df.loc[idx]
            if not any(rule.cover(neg_ex) for rule in rules):
                return neg_ex
        return df.loc[df.sample(n=1).index[0]]

    def _generate_stars(self, seed, negatives, predicate) -> set[Rule]:
        def examples_covered_rate(star: Rule):
            return negatives.apply(lambda x: star.cover(x), axis=1).mean()

        partial_stars: set[Rule] = set()
        i = 0
        work_df = negatives
        count_log = []
        while len(work_df) > 0 and i < self.max_star_it:
            neg_ex = self._rand_not_covered(partial_stars, work_df)
            # Get (column, value) that rule cannot have to not cover seed
            diff_columns = neg_ex[seed != neg_ex]

            if len(partial_stars) == 0:
                for ps in generate_partial_stars(diff_columns, neg_ex.index, predicate):
                    ps.covered = examples_covered_rate(ps)
                    partial_stars.add(ps)
                continue

            # Modify present stars with new values for column
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
            count_log.append(len(work_df))
            
            if len(count_log) > 20:
                if len(set(count_log[-20:])) <=1:
                    return partial_stars
            i += 1
        return partial_stars

def perform_experiment(exp_data):
    result = -1
    for _ in range(3):
        df = pd.read_csv(exp_data['df_path'])
        df['y'] = df['y'].map(lambda x: x == exp_data['mapping'])
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='y'), df['y'], test_size=0.2,random_state=2137)
        clf = AQClassifier(**exp_data)
        res = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Accuracy: {classification_report(y_test, y_pred)}")
        if accuracy_score(y_test, y_pred) > result:
            result = accuracy_score(y_test, y_pred)
            clf.save(exp_data['filename'])
            with open(exp_data['result_path'], 'w') as fh:
                json.dump(res, fh)

def main(experiments):
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as exc:
        jobs = [exc.submit(perform_experiment, exp_data) for exp_data in experiments]
        for job in jobs:
            job.result()
        

experiments = [
    {
        'star_it': 70,
        'it': 50,
        'max_cpx': 10,
        'max_rules': 15,
        'no_diff_it': 5,
        'df_path': "data/processed/adult.csv",
        'filename': "models/aq_adult_50_50_10_15_10",
        "result_path": "models/aq_adult_50_50_10_15_10_res.csv",
        'mapping': " <=50K"
    },
    {
        'star_it': 200,
        'it': 50,
        'max_cpx': 400,
        'max_rules': 15,
        'no_diff_it': 10,
        'df_path': "data/processed/bank-full.csv",
        'filename': "models/aq_bank_50_50_10_15_10",
        "result_path": "models/aq_bank_50_50_10_15_10_res.json",
        'mapping': "yes"
    },
]
  
if __name__ == '__main__':
    main(experiments)
    

    
