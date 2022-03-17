import string
import random
from collections import defaultdict
from typing import DefaultDict, Dict, Set, FrozenSet, Type
from itertools import product


CHARSET = string.digits[:5]
MAX_LENGTH = 3
MAX_TURNS = 8


class Checker:
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    @staticmethod
    def check(target: str, guess: str) -> str:
        # B for black, G for Green, Y for yellow

        assert len(target) == len(guess), f"{target}, {guess} has different length."

        pattern = ""
        for idx, c in enumerate(guess):
            if target[idx] == c:
                pattern += "G"
            elif c in target:
                pattern += "Y"
            else:
                assert c not in target
                pattern += "B"
        return pattern

    def is_success(self, pattern) -> bool:
        return pattern == "G" * self.max_length


class ABChecker(Checker):
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    @staticmethod
    def check(target: str, guess: str) -> str:

        assert len(target) == len(guess)

        count_a = count_b = 0
        for idx, c in enumerate(guess):
            if target[idx] == c:
                count_a += 1
            elif c in target:
                count_b += 1
        return f"{count_a}A{count_b}B"

    def is_success(self, pattern) -> bool:
        return pattern == f"{self.max_length}A0B"


class Solver:
    def __init__(
        self,
        checker: Checker,
        full_set: FrozenSet[str],
        max_length: int,
        available: Set[str] = set(),
    ) -> None:
        self.max_length = max_length
        self.checker = checker
        self.full_set = full_set
        if len(available) > 0:
            self.available = available
        else:
            self.available = set(full_set.copy())

    def guess(self) -> str:
        return min(self.available)

    def update(self, guess: str, pattern: str):
        available = set()
        for target in self.available:
            if self.checker.check(target, guess) == pattern:
                available.add(target)
        self.available = available


class RandomSolver(Solver):
    def guess(self) -> str:
        return random.choice(tuple(self.available))


class Evaluator:
    def __init__(
        self, checker: Checker, solver_type: Type[Solver], max_length: int, charset: str
    ) -> None:
        self.checker = checker
        self.solver_type = solver_type
        self.max_length = max_length
        self.full_set = frozenset(
            "".join(c) for c in product(charset, repeat=max_length)
        )

    def evaluate(self):
        results = []
        for idx, target in enumerate(self.full_set):
            # print(f"========================={idx}, {target}=========================")
            solver = self.solver_type(self.checker, self.full_set, self.max_length)

            current = 0
            while True:
                current += 1
                # print(f"Round {current}: {len(solver.available)}")
                guess = solver.guess()
                pattern = self.checker.check(target, guess)
                # print(f"Guess: {guess}, Pattern: {pattern}")

                if self.checker.is_success(pattern):
                    break
                else:
                    solver.update(guess, pattern)

                if current > 100:
                    break

            results.append(current)

        assert not 101 in results
        print(f"Total: {sum(results)}")
        print(f"Avg: {sum(results) / len(results)}")
        print(f"Max: {max(results)}")


class MinMaxPatternSolver(Solver):
    def guess(self) -> str:

        if len(self.available) == 1:
            return self.available.pop()

        results: Dict[str, DefaultDict[str, set]] = {}

        # for guess in self.available:
        for guess in self.full_set:
            results[guess] = defaultdict(set)
            for target in self.available:
                pattern = self.checker.check(target, guess)
                results[guess][pattern].add(target)

        guess_pattern_count = []
        for guess, patterns in results.items():
            max_pattern_count = 0
            for pattern, pattern_availables in patterns.items():
                max_pattern_count = max(max_pattern_count, len(pattern_availables))

            guess_pattern_count.append((guess, max_pattern_count))

        best_guess = min(guess_pattern_count, key=lambda x: x[1])[0]

        return best_guess


class OptimalSolver(Solver):

    decision: Dict[str, dict] = {'132': {'BBB': {'000': {'GGG': {}, 'GGY': {'004': {'GGG': {}}}, 'GYG': {'040': {'GGG': {}}}, 'GYY': {'044': {'GGG': {}}}, 'YGG': {'400': {'GGG': {}}}, 'YGY': {'404': {'GGG': {}}}, 'YYG': {'440': {'GGG': {}}}, 'BBB': {'444': {'GGG': {}}}}}, 'YBB': {'014': {'GYB': {'001': {'GGG': {}}}, 'GGB': {'010': {'GGG': {}, 'GGY': {'011': {'GGG': {}}}}}, 'GGG': {}, 'GYY': {'041': {'GGG': {}}}, 'YYY': {'401': {'GGG': {}}}, 'YGY': {'410': {'GGG': {}}}, 'BGY': {'411': {'GGG': {}}}, 'BGG': {'414': {'GGG': {}}}, 'BYY': {'441': {'GGG': {}}}}}, 'BBG': {'024': {'GYB': {'002': {'GGG': {}}}, 'GGB': {'022': {'GGG': {}}}, 'GYY': {'042': {'GGG': {}}}, 'YYB': {'202': {'GGG': {}}}, 'BGB': {'222': {'GGG': {}}}, 'BYY': {'242': {'GGG': {}, 'YGG': {'442': {'GGG': {}}}}}, 'YYY': {'402': {'GGG': {}}}, 'BGY': {'422': {'GGG': {}}}}}, 'BYB': {'043': {'GBG': {'003': {'GGG': {}}}, 'GGG': {}, 'YBY': {'300': {'GGG': {}}}, 'YBG': {'303': {'GGG': {}}}, 'YYY': {'304': {'GGG': {}}}, 'YGY': {'340': {'GGG': {}}}, 'BGG': {'343': {'GGG': {}, 'YGG': {'443': {'GGG': {}}}}}, 'BGY': {'344': {'GGG': {}}}, 'YYG': {'403': {'GGG': {}}}}}, 'YBG': {'004': {'GYB': {'012': {'GGG': {}}}, 'BBB': {'212': {'GGG': {}}}, 'BBY': {'412': {'GGG': {}}}}}, 'YYB': {'014': {'GGB': {'013': {'GGG': {}}}, 'YYB': {'301': {'GGG': {}}}, 'YGB': {'310': {'GGG': {}}}, 'BGB': {'311': {'GGG': {}, 'GGY': {'313': {'GGG': {}}}}}, 'BGG': {'314': {'GGG': {}}}, 'BYY': {'341': {'GGG': {}}}, 'BGY': {'413': {'GGG': {}}}}}, 'BBY': {'024': {'GGB': {'020': {'GGG': {}}}, 'GGG': {}, 'YYB': {'200': {'GGG': {}}}, 'YYG': {'204': {'GGG': {}}}, 'YGB': {'220': {'GGG': {}}}, 'BGG': {'224': {'GGG': {}, 'YGG': {'424': {'GGG': {}}}}}, 'YYY': {'240': {'GGG': {}}}, 'BYG': {'244': {'GGG': {}}}, 'YGY': {'420': {'GGG': {}}}}}, 'YBY': {'014': {'GYB': {'021': {'GGG': {}}}, 'YYB': {'201': {'GGG': {}}}, 'YGB': {'210': {'GGG': {}}}, 'BGB': {'211': {'GGG': {}}}, 'BGG': {'214': {'GGG': {}}}, 'BYB': {'221': {'GGG': {}}}, 'BYY': {'241': {'GGG': {}, 'YYG': {'421': {'GGG': {}}}}}}}, 'BYY': {'024': {'GGB': {'023': {'GGG': {}}}, 'YYB': {'203': {'GGG': {}}}, 'BGB': {'223': {'GGG': {}, 'YGG': {'323': {'GGG': {}}}}}, 'BYY': {'243': {'GGG': {}}}, 'YGB': {'320': {'GGG': {}}}, 'BGG': {'324': {'GGG': {}}}, 'BGY': {'423': {'GGG': {}}}}}, 'BGB': {'043': {'GBY': {'030': {'GGG': {}}}, 'GBG': {'033': {'GGG': {}}}, 'GYY': {'034': {'GGG': {}}}, 'YBY': {'330': {'GGG': {}}}, 'BBG': {'333': {'GGG': {}}}, 'BYY': {'334': {'GGG': {}, 'YGG': {'434': {'GGG': {}}}}}, 'YYY': {'430': {'GGG': {}}}, 'BYG': {'433': {'GGG': {}}}}}, 'YGB': {'004': {'GYB': {'031': {'GGG': {}}}, 'BBB': {'331': {'GGG': {}}}, 'BBY': {'431': {'GGG': {}}}}}, 'BGG': {'204': {'YYB': {'032': {'GGG': {}}}, 'GBB': {'232': {'GGG': {}}}, 'YBB': {'332': {'GGG': {}}}, 'YBY': {'432': {'GGG': {}}}}}, 'GBB': {'014': {'YYB': {'100': {'GGG': {}, 'GGY': {'101': {'GGG': {}}}}}, 'YYG': {'104': {'GGG': {}}}, 'YGB': {'110': {'GGG': {}}}, 'BGB': {'111': {'GGG': {}}}, 'BGG': {'114': {'GGG': {}}}, 'YYY': {'140': {'GGG': {}}}, 'BYY': {'141': {'GGG': {}}}, 'BYG': {'144': {'GGG': {}}}}}, 'GBG': {'014': {'YYB': {'102': {'GGG': {}}}, 'BGB': {'112': {'GGG': {}}}, 'BYB': {'122': {'GGG': {}}}, 'BYY': {'142': {'GGG': {}}}}}, 'GYB': {'004': {'YGB': {'103': {'GGG': {}}}, 'BBB': {'113': {'GGG': {}}}, 'BBY': {'143': {'GGG': {}}}}}, 'GBY': {'001': {'YYY': {'120': {'GGG': {}}}, 'BBG': {'121': {'GGG': {}}}, 'BBY': {'124': {'GGG': {}}}}}, 'GYY': {'123': {'GGG': {}}}, 'GGB': {'041': {'YBY': {'130': {'GGG': {}}}, 'BBG': {'131': {'GGG': {}}}, 'BBY': {'133': {'GGG': {}}}, 'BYY': {'134': {'GGG': {}}}}}, 'GGG': {}, 'YYY': {'213': {'GGG': {}, 'YYY': {'321': {'GGG': {}}}}}, 'BGY': {'003': {'YYY': {'230': {'GGG': {}}}, 'BBG': {'233': {'GGG': {}}}, 'BBY': {'234': {'GGG': {}}}}}, 'YGY': {'231': {'GGG': {}}}, 'BYG': {'004': {'YGB': {'302': {'GGG': {}}}, 'BBB': {'322': {'GGG': {}}}, 'BBY': {'342': {'GGG': {}}}}}, 'YYG': {'312': {'GGG': {}}}}}

    def guess(self) -> str:
        return list(self.decision.keys())[0]

    def update(self, guess: str, pattern: str):
        self.decision = self.decision[guess][pattern]


def get_pattern_for_a_guess(guess: str, checker: Checker, available: Set[str]):
    results = defaultdict(set)
    for target in available:
        pattern = checker.check(target, guess)
        results[pattern].add(target)
    return results


def main():
    # checker = ABChecker(MAX_LENGTH)
    checker = Checker(MAX_LENGTH)

    # evaluator = Evaluator(checker, Solver, MAX_LENGTH, CHARSET)
    # evaluator = Evaluator(checker, RandomSolver, MAX_LENGTH, CHARSET)
    # evaluator = Evaluator(checker, MinMaxPatternSolver, MAX_LENGTH, CHARSET)
    evaluator = Evaluator(checker, OptimalSolver, MAX_LENGTH, CHARSET)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
