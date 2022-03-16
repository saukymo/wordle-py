import string
import random
from collections import defaultdict
from typing import DefaultDict, Dict, Set, FrozenSet, Type
from itertools import product


CHARSET = string.digits[:4]
MAX_LENGTH = 3
MAX_TURNS = 8

class Checker:
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    @staticmethod
    def check(target: str, guess: str) -> str:
        # B for black, G for Green, Y for yellow

        assert len(target) == len(guess), f'{target}, {guess} has different length.'

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
            print(f"========================={idx}, {target}=========================")
            solver = self.solver_type(self.checker, self.full_set, self.max_length)

            current = 0
            while True:
                current += 1
                print(f"Round {current}: {len(solver.available)}")
                guess = solver.guess()
                pattern = self.checker.check(target, guess)
                print(f"Guess: {guess}, Pattern: {pattern}")

                if self.checker.is_success(pattern):
                    break
                else:
                    solver.update(guess, pattern)

                if current > 100:
                    break

            results.append(current)
            if idx > 100:
                break

        assert not 101 in results
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

    decision: Dict[str, dict] = {'231': {'GGG': {}, 'GBB': {'220': {'GGG': {}, 'GGB': {'222': {'GGG': {}}}, 'GYG': {'200': {'GGG': {}}}, 'GYY': {'202': {'GGG': {}}}}}, 'YBY': {'012': {'YYG': {'102': {'GGG': {}}}, 'BYG': {'122': {'GGG': {}}}, 'GGG': {}, 'YYY': {'120': {'GGG': {}}}, 'BGG': {'112': {'GGG': {}}}}}, 'BYY': {'310': {'GGG': {}, 'YYY': {'103': {'GGG': {}}}, 'YGY': {'013': {'GGG': {}}}, 'YGB': {'113': {'GGG': {}}}, 'GGB': {'313': {'GGG': {}}}}}, 'BBY': {'100': {'GGG': {}, 'YYG': {'010': {'GGG': {}}}, 'GYG': {'110': {'GGG': {}}}}}, 'GYY': {'213': {'GGG': {}}}, 'BGB': {'033': {'GGG': {}, 'BGG': {'333': {'GGG': {}}}, 'YGY': {'330': {'GGG': {}}}, 'GGY': {'030': {'GGG': {}}}}}, 'BGG': {'131': {'GGG': {}, 'YGG': {'331': {'GGG': {}, 'YGG': {'031': {'GGG': {}}}}}}}, 'GBY': {'212': {'GGG': {}, 'GGY': {'210': {'GGG': {}}}}}, 'BBG': {'101': {'GGG': {}, 'YGG': {'001': {'GGG': {}}}, 'GBG': {'111': {'GGG': {}}}, 'YYG': {'011': {'GGG': {}}}}}, 'YYB': {'023': {'GGG': {}, 'YGY': {'320': {'GGG': {}}}, 'YYY': {'302': {'GGG': {}}}, 'BGY': {'322': {'GGG': {}}}, 'BGG': {'323': {'GGG': {}}}}}, 'GYB': {'223': {'GGG': {}, 'GYG': {'203': {'GGG': {}}}}}, 'BYG': {'311': {'GGG': {}, 'GYG': {'301': {'GGG': {}}}}}, 'YBB': {'002': {'GGG': {}, 'GYG': {'022': {'GGG': {}}}, 'GYY': {'020': {'GGG': {}}}}}, 'BYB': {'003': {'GGG': {}, 'YGG': {'303': {'GGG': {}}}, 'YGY': {'300': {'GGG': {}}}}}, 'YYY': {'312': {'GGG': {}, 'YYY': {'123': {'GGG': {}}}}}, 'YBG': {'121': {'GGG': {}, 'YGG': {'021': {'GGG': {}}}}}, 'BGY': {'133': {'GGG': {}, 'GGY': {'130': {'GGG': {}}}}}, 'YGB': {'032': {'GGG': {}, 'BGG': {'332': {'GGG': {}}}}}, 'GGB': {'233': {'GGG': {}, 'GGY': {'232': {'GGG': {}, 'GGY': {'230': {'GGG': {}}}}}}}, 'YGY': {'132': {'GGG': {}}}, 'GBG': {'211': {'GGG': {}, 'GYG': {'201': {'GGG': {}, 'GBG': {'221': {'GGG': {}}}}}}}, 'YYG': {'321': {'GGG': {}}}, 'BBB': {'000': {'GGG': {}}}}}

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


def solve(
    current: int,
    available: Set[str],
    checker: Checker,
    full_set: FrozenSet[str],
    attempts: list,
):

    assert len(available) > 0

    if current > MAX_TURNS:
        return

    print(
        f"=========================LeveL: {current},  No. of possibilities: {len(available)}========================="
    )

    if len(available) == 1:
        guess = available.pop()
        return ({guess: "GG"}, {guess: 1})

    results: Dict[str, DefaultDict[str, set]] = {}
    for guess in available:
        results[guess] = defaultdict(set)
        for target in available:
            pattern = checker.check(target, guess)
            results[guess][pattern].add(target)

    for guess, patterns in results.items():
        count = len(patterns.keys())

        max_pattern_count = 0
        for pattern, pattern_available in patterns.items():
            max_pattern_count = max(max_pattern_count, len(pattern_available))

        print(
            f"For {guess}, there are {count} kinds of patterns while the largetest one is {max_pattern_count}"
        )
        print(patterns)

    # DFS next level
    decisions = {}
    print("Attempts: ", attempts)

    records: Dict[str, dict] = {}
    for guess, patterns in results.items():
        # 初始值改成 1， 因为还有当前猜的这一次。
        counter = {}
        for a in available:
            counter[a] = 1

        records[guess] = {}

        for pattern, pattern_available in patterns.items():
            # 如果是GG，游戏结束
            if checker.is_success(pattern):
                counter[guess] += 0
                records[guess][pattern] = "GG"
                continue

            print(
                f"---------------------------{guess}, {pattern}---------------------------"
            )
            print(pattern_available)
            print(
                f'---------------------------{"-"*(2 * 2 + 2)}---------------------------'
            )

            # 每个pattern_available对应的决策树和每个结果的猜测次数
            (best_guesses, guess_numbers) = solve(
                current + 1,
                pattern_available,
                checker,
                full_set,
                attempts + [(guess, pattern)],
            )
            print(
                f"LeveL: {current}, Guess: {guess}, Pattern: {pattern}, Best Guess: {best_guesses}: ",
                guess_numbers,
            )

            records[guess][pattern] = best_guesses

            for target, count in guess_numbers.items():
                counter[target] += count

        print(
            f"*Level: {current}, Guess {guess}, {patterns}, Attempts: {attempts}: ",
            counter,
        )
        decisions[guess] = counter

    print(f"Available: ", available)
    for a in available:
        print(f"{a} -> ", results[a])
    print(f"LeveL: {current}, Decisions: ", decisions)

    min_avg = 10000000.0
    best_guess = None
    for guess, counter in decisions.items():
        avg = sum(counter.values()) / len(counter.values())
        if avg < min_avg:
            min_avg = avg
            best_guess = guess
        print(f"Guess {guess}, Avg {avg}")

    assert best_guess is not None
    return ({best_guess: records[best_guess]}, decisions[best_guess])


def main():
    # checker = ABChecker(MAX_LENGTH)
    checker = Checker(MAX_LENGTH)

    # full_set = frozenset(
    #         "".join(c) for c in product(CHARSET, repeat=MAX_LENGTH)
    #     )
    # print(solve(0, set(full_set), checker, full_set, []))

    # evaluator = Evaluator(checker, Solver, MAX_LENGTH, CHARSET)
    # evaluator = Evaluator(checker, RandomSolver, MAX_LENGTH, CHARSET)
    # evaluator = Evaluator(checker, MinMaxPatternSolver, MAX_LENGTH, CHARSET)
    evaluator = Evaluator(checker, OptimalSolver, MAX_LENGTH, CHARSET)
    evaluator.evaluate()


if __name__ == "__main__":
    # main()

    checker = ABChecker(4)
    for target in ['1111', '1112', '1121', '1122']:
        print(target, checker.check(target, '1112'))
