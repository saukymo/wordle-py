import string
import random
from collections import defaultdict
from typing import DefaultDict, Dict, Set, FrozenSet, Type
from itertools import product


CHARSET = string.digits[:4]
MAX_LENGTH = 3


class Checker:
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    @staticmethod
    def check(target: str, guess: str) -> str:
        # B for black, G for Green, Y for yellow

        assert len(target) == len(guess)

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

    decision: Dict[str, dict] = {
        "031": {
            "BGY": {"132": {"GGG": "GG", "GGB": {"133": "GG"}}},
            "GGG": "GG",
            "YYB": {
                "203": {
                    "GGG": "GG",
                    "YGY": {"302": "GG"},
                    "BGY": {"300": "GG"},
                    "BGG": {"303": "GG"},
                    "YYY": {"320": "GG"},
                }
            },
            "GBG": {
                "001": {
                    "GGG": "GG",
                    "GYG": {"011": {"GGG": "GG", "GYG": {"021": "GG"}}},
                }
            },
            "GBB": {
                "022": {
                    "GGG": "GG",
                    "GGY": {"020": "GG"},
                    "GBB": {"000": "GG"},
                    "GYG": {"002": "GG"},
                }
            },
            "YBY": {
                "210": {
                    "BYG": {"100": "GG"},
                    "YYG": {"120": "GG"},
                    "GGG": "GG",
                    "YYY": {"102": "GG"},
                    "BGG": {"110": "GG"},
                }
            },
            "YYG": {"301": "GG"},
            "BBY": {"122": {"GGG": "GG", "YYG": {"212": "GG"}, "GYG": {"112": "GG"}}},
            "YBB": {"220": {"GGG": "GG", "GYY": {"202": "GG"}, "GYG": {"200": "GG"}}},
            "YGB": {"230": {"GGG": "GG", "BGG": {"330": "GG"}}},
            "BYB": {"323": {"GGG": "GG", "YGG": {"223": "GG"}, "GGY": {"322": "GG"}}},
            "BGB": {
                "232": {
                    "GGG": "GG",
                    "BGB": {"333": "GG"},
                    "GGY": {"233": "GG"},
                    "YGG": {"332": "GG"},
                }
            },
            "GGB": {
                "032": {
                    "GGG": "GG",
                    "GGB": {"033": {"GGG": "GG", "GGY": {"030": "GG"}}},
                }
            },
            "BYY": {
                "312": {
                    "GGG": "GG",
                    "YGY": {"213": "GG"},
                    "YYY": {"123": "GG"},
                    "GGB": {"313": "GG"},
                    "YGB": {"113": "GG"},
                }
            },
            "YBG": {"101": {"GGG": "GG", "YGG": {"201": "GG"}}},
            "GYY": {"013": "GG"},
            "YGY": {"130": "GG"},
            "BYG": {"321": {"GGG": "GG", "GBG": {"311": "GG"}}},
            "BGG": {
                "331": {
                    "GGG": "GG",
                    "YGG": {"231": {"GGG": "GG", "BGG": {"131": "GG"}}},
                }
            },
            "BBG": {
                "121": {
                    "GGG": "GG",
                    "GBG": {"111": "GG"},
                    "YYG": {"211": "GG"},
                    "YGG": {"221": "GG"},
                }
            },
            "YYY": {"310": {"GGG": "GG", "YYY": {"103": "GG"}}},
            "GBY": {"010": {"GGG": "GG", "GGY": {"012": "GG"}}},
            "GYB": {"023": {"GGG": "GG", "GBG": {"003": "GG"}}},
            "BBB": {"222": "GG"},
        }
    }

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

            # 每个pattern_available对应的平均猜测次数
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
    main()
