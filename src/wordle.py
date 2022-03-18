import json
from src.main import Checker, Solver
from itertools import product
from typing import DefaultDict, Dict, Set, FrozenSet, List, Type

class WordleSolver:

    def __init__(self):
        self.decision = json.load(open('results/wordle.json'))

    def guess(self) -> str:
        return list(self.decision.keys())[0]

    def update(self, guess: str, pattern: str):
        self.decision = self.decision[guess][pattern]


class Evaluator:
    def __init__(
        self, checker: Checker, solver_type: Type[WordleSolver], answers: List[str], full_set: List[str]
    ) -> None:
        self.checker = checker
        self.solver_type = solver_type
        self.answers = answers
        self.full_set = full_set

    def evaluate(self):
        results = []
        for idx, target in enumerate(self.answers):
            # print(f"========================={idx}, {target}=========================")
            solver = self.solver_type()

            current = 0
            while True:
                current += 1
                # print(f"Round {current}")
                guess = solver.guess()
                pattern = self.checker.check(target, guess)

                # if current >= 8:
                #     print(f"Guess: {guess}, Pattern: {pattern}")

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

if  __name__ == '__main__':
    checker = Checker(5)
    answers = open('data/answers.txt').read().split('\n')
    assert len(answers) == 2315

    wordlist = open('data/wordlist.txt').read().split('\n')
    assert len(wordlist) == 12972

    evaluator = Evaluator(checker, WordleSolver, answers, wordlist)
    evaluator.evaluate()