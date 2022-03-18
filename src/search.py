import json
import string
from math import log2, floor
from itertools import product
from collections import defaultdict
from typing import DefaultDict, Tuple, Dict, List

from src.main import Checker

CHARSET = string.digits[:4]
MAX_LENGTH = 5
MAX_TURNS = 7


total = 0

MAX_CACHE_COUNT = 9
cache: Dict[str, Tuple[dict, int, int]] = {}

def dfs(
    current: int, availables: List[str], checker: Checker, full_set: List[str]
) -> Tuple[dict, int, int]:

    # 如果猜测可能是0信息量的话，必须要这个条件，因为一直无信息量可能死循环
    # if current > MAX_TURNS:
    #     return {}, -1, -1

    sorted_availables = sorted(availables)

    # 记忆化，对于搜索过的availables，我们可以直接得到结果，而不用搜索
    # 对于 7 charset 和 3 max length, max_cache_count = 5 的情况下
    # 不加记忆化需要迭代110181次，加上之后迭代1839次

    # NOTE: 如果限制了最大搜索深度，cache也需要限制最大深度
    # NOTE: 合并状态，检查同构，并从cache里恢复decision tree 
    # 合并状态可能没有办法做，因为wordle限制了猜测要是合法单词，
    # 导致特定状态下的最优猜测可能不得不包含更多字母，那么当其他状态同构的时候，
    # 可能导致多出来的单词不合法。
    # 所以状态合并的前提还是只能猜测availables，然而这个已经被证明不论是什么模式
    # 都不成立

    if len(availables) <= MAX_CACHE_COUNT:
        key = ''.join(sorted_availables)
        if key in cache:
            return cache[key]
            
    # NOTE: 即使是hard mode, 搜索集也不应该是availables，full_set里面仍然会有满足限制的结果
    # 如果只剩2个数，2选1即可，没有必要检查fullset
    # 对于3个数，最好情况是一次猜测，把结果三分然后再猜一次，这样平均每个结果2次。但是直接猜其中一个，平均每个结果也是2次(1+2+3), 所以3个结果也可以3选1
    # 对于 5 charset 和 4 max length, > 2 的迭代次数是390277, >3的迭代次数是152197
    # valid_guess = full_set if len(availables) > 3 else availables
    valid_guess = availables

    if current == 0:
        valid_guess = ['salet']
    
    pattern_results: Dict[str, DefaultDict[str, list]] = {}
    full_set_groups: Dict[str, DefaultDict[str, list]] = {}
    entropy_results = []
        
    # 对于 5 charset 和 2 max length, full_set 平均猜测次数是 3.0 而 availables 的平均猜测次数是 3.08
    for guess in valid_guess:

        # 根据 pattern 对 available 分组
        pattern_results[guess] = defaultdict(list)
        full_set_groups[guess] = defaultdict(list)
        # for target in sorted_availables:
        for target in sorted(full_set):
            pattern = checker.check(target, guess)
            full_set_groups[guess][pattern].append(target)

            if target in sorted_availables:
                pattern_results[guess][pattern].append(target)
            
        # 计算当前 guess 的期望信息熵
        entropy: float = 0
        for pattern, pattern_availables in pattern_results[guess].items():
            entropy += len(pattern_availables) * log2(len(pattern_availables)) 
        entropy_results.append((guess, entropy))

    # TODO: 对于 current == 1 时，可以把 pattern 和 对应的 availables 分给不同的进程搜索

    best_guess = None
    best_guess_count = -1
    best_decision_tree = None
    max_level = -1

    # 对于 3 charset 和 3 max length, 不排序需要迭代2038次，排序后迭代2009次
    for guess, entropy in sorted(entropy_results, key=lambda x: x[1]):
        # for guess, entropy in entropy_results:

        # entropy 意味着每次二分的猜测次数。如果这样的猜测次数都超过了当前最优解，那么跳过
        # 对于 3 charset 和 3 max length, 去掉这个优化之前是2009次迭代，加上后只需要343次迭代
        # FIXME: 这个优化会剪掉最优解, 因为最好情况不是二分，而是K分，对于wordle来说，K=3^5=243
        if (
            best_guess_count != -1
            and len(availables) + floor(entropy) > best_guess_count
        ):
            continue

        # 如果当前guess不能有效缩小available size，跳过
        if len(pattern_results[guess].keys()) == 1:
            continue

        # guess_count_sum 是对于当前这个 guess，剩下所有可能结果的猜测次数之和, 而每个结果都至少还要猜1次，后面的post_guess_count是不含本次猜测结果的。
        guess_count_sum = len(availables)
        decision_tree = {}

        post_levels = []
        for idx, (pattern, pattern_availables) in enumerate(pattern_results[guess].items()):
            # 如果当前解就是答案，不用继续猜了。
            if checker.is_success(pattern):
                post_decision_tree: Dict[str, dict] = {}
                post_guess_count = 0
                max_post_level = 0
            # 如果还剩唯一解，可以直接猜
            elif len(pattern_availables) == 1:
                next_guess = pattern_availables[0]
                post_decision_tree = {next_guess: {"GGGGG": {}}}
                post_guess_count = 1
                max_post_level = 1
            else:
                post_decision_tree = {}
                post_guess_count = -1
                max_post_level = - 1

                # 只有hard mode需要限制fullset
                # 两个返回值，best_guesses 是当前状态的最优决策树，guess_count 是对于这棵决策树来说 pattern_availables 里每个可行解的猜测次数
                (post_decision_tree, post_guess_count, max_post_level) = dfs(
                    current + 1, pattern_availables, checker, full_set_groups[guess][pattern]
                )

                # 如果当前pattern无解，说明这个guess无解，那么就不用继续检查了
                if post_decision_tree == {}:
                    break

            # 如果检查过的pattern的猜测次数已经超过当前最优解了，那么可以跳过这个guess
            if best_guess_count != -1 and (guess_count_sum > best_guess_count):
                break

            decision_tree[pattern] = post_decision_tree
            guess_count_sum += post_guess_count
            post_levels.append(max_post_level)

        # 进入 else 分支说明所有pattern都是valid
        else:
            assert idx == len(pattern_results[guess].keys()) - 1

            # 如果当前 guess 优于历史值，更新记录
            if best_guess_count == -1 or guess_count_sum < best_guess_count:
                best_guess_count = guess_count_sum
                best_guess = guess
                best_decision_tree = decision_tree
                max_level = max(post_levels)

    # 由于强剪枝，有无解的可能
    # if best_guess is None:
    #     return {}, -1, -1
    assert best_guess is not None

    best_result = ({best_guess: best_decision_tree}, best_guess_count, max_level + 1)

    if len(availables) <= MAX_CACHE_COUNT:
        # For debug
        cache[''.join(sorted_availables)] = best_result

    # 返回最优解
    return best_result


def main():
    checker = Checker(MAX_LENGTH)

    full_set = ["".join(c) for c in product(CHARSET, repeat=MAX_LENGTH)]

    assert all(len(x) == MAX_LENGTH for x in full_set)

    answers = open('data/answers.txt').read().split('\n')
    assert len(answers) == 2315

    wordlist = open('data/wordlist.txt').read().split('\n')
    assert len(wordlist) == 12972

    decision_tree, post_guess_count, max_level = dfs(0, answers, checker, wordlist)
    json.dump(decision_tree, open(f'results/wordle.json', 'w'))

    print('Total:', post_guess_count)
    print('Avg:', post_guess_count / len(answers))
    print("Iterations:", total)
    print("Max Level:", max_level)

    # decision_tree, post_guess_count = dfs(0, full_set, checker, full_set)
    # json.dump(decision_tree, open(f'results/l{MAX_LENGTH}c{len(CHARSET)}.json', 'w'))

    # print('Total:', post_guess_count)
    # print('Avg:', post_guess_count / len(full_set))
    # print("Iterations:", total)


if __name__ == "__main__":
    main()
