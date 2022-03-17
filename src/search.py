import json
import string
from math import log2, ceil
from itertools import product
from collections import defaultdict
from typing import DefaultDict, Tuple, Dict, List

from src.main import Checker

CHARSET = string.digits[:5]
MAX_LENGTH = 2
MAX_TURNS = 5


total = 0


def dfs(
    current: int, availables: List[str], checker: Checker, full_set: List[str]
) -> Tuple[dict, int]:
    # TODO：记忆化，对于搜索过的availables，我们可以直接得到结果，而不用搜索
    # TODO：如果能计算出两步能够得到的全集，那么如果当前状态不在集合中，即可比较max_turns，然后过滤。

    # 如果猜测集是full_set的话，必须要这个条件，因为full_set里会有大量没有信息量的guess
    if current > MAX_TURNS:
        return {}, -1

    global total
    total += 1

    pattern_results: Dict[str, DefaultDict[str, list]] = {}
    entropy_results = []

    # 如果只剩2个数，2选1即可，没有必要检查fullset
    # 对于3个数，最好情况是一次猜测，把结果三分然后再猜一次，这样平均每个结果2次。但是直接猜其中一个，平均每个结果也是2次(1+2+3), 所以3个结果也可以3选1
    # 对于 5 charset 和 4 max length, > 2 的迭代次数是390277, >3的迭代次数是152197
    valid_guess = full_set if len(availables) > 3 else availables

    # 对于 5 charset 和 2 max length, full_set 平均猜测次数是 3.0 而 availables 的平均猜测次数是 3.08
    for guess in valid_guess:

        # 根据 pattern 对 available 分组
        pattern_results[guess] = defaultdict(list)
        for target in sorted(availables):
            pattern = checker.check(target, guess)
            pattern_results[guess][pattern].append(target)

        # 计算当前 guess 的期望信息熵
        entropy: float = 0
        for pattern, pattern_availables in pattern_results[guess].items():
            entropy += len(pattern_availables) * log2(len(pattern_availables))
        entropy_results.append((guess, entropy))

    best_guess = None
    best_guess_count = -1
    best_decision_tree = None

    # 对于 3 charset 和 3 max length, 不排序需要迭代2038次，排序后迭代2009次
    for guess, entropy in sorted(entropy_results, key=lambda x: x[1]):
        # for guess, entropy in entropy_results:

        # entropy 意味着每次二分的猜测次数。如果这样的猜测次数都超过了当前最优解，那么跳过
        # 对于 3 charset 和 3 max length, 去掉这个优化之前是2009次迭代，加上后只需要343次迭代
        if (
            best_guess_count != -1
            and len(availables) + ceil(entropy) >= best_guess_count
        ):
            continue

        # 如果当前guess不能有效缩小available size，跳过
        if len(pattern_results[guess].keys()) == 1:
            continue

        # guess_count_sum 是对于当前这个 guess，剩下所有可能结果的猜测次数之和, 而每个结果都至少还要猜1次，后面的post_guess_count是不含本次猜测结果的。
        guess_count_sum = len(availables)
        decision_tree = {}

        for pattern, pattern_availables in pattern_results[guess].items():
            # 如果当前解就是答案，不用继续猜了。
            if checker.is_success(pattern):
                post_decision_tree: Dict[str, dict] = {}
                post_guess_count = 0
            # 如果还剩唯一解，可以直接猜
            elif len(pattern_availables) == 1:
                next_guess = pattern_availables[0]
                post_decision_tree = {next_guess: {"GGG": {}}}
                post_guess_count = 1
            else:
                # 两个返回值，best_guesses 是当前状态的最优决策树，guess_count 是对于这棵决策树来说 pattern_availables 里每个可行解的猜测次数
                (post_decision_tree, post_guess_count) = dfs(
                    current + 1, pattern_availables, checker, full_set
                )

                # 如果当前pattern无解，说明这个guess无解，那么就不用继续检查了
                if post_decision_tree == None:
                    break

            decision_tree[pattern] = post_decision_tree
            guess_count_sum += post_guess_count

            # 如果检查过的pattern的猜测次数已经超过当前最优解了，那么可以跳过这个guess
            if best_guess_count != -1 and (guess_count_sum > best_guess_count):
                break

        # 进入 else 分支说明所有pattern都是valid
        else:
            # 如果当前 guess 优于历史值，更新记录
            if best_guess_count == -1 or guess_count_sum < best_guess_count:
                best_guess_count = guess_count_sum
                best_guess = guess
                best_decision_tree = decision_tree

    # 由于强剪枝，有无解的可能
    if best_guess is None:
        return {}, -1

    # 返回最优解
    return {best_guess: best_decision_tree}, best_guess_count


def main():
    checker = Checker(MAX_LENGTH)

    full_set = ["".join(c) for c in product(CHARSET, repeat=MAX_LENGTH)]

    assert all(len(x) == MAX_LENGTH for x in full_set)

    decision_tree, post_guess_count = dfs(0, full_set, checker, full_set)

    print(decision_tree)

    json.dump(decision_tree, open(f'results/l{MAX_LENGTH}c{len(CHARSET)}.json', 'w'))

    print('Total:', post_guess_count)
    print('Avg:', post_guess_count / len(full_set))
    print("Total:", total)


if __name__ == "__main__":
    main()
