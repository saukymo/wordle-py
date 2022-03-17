import string
from itertools import product
from collections import defaultdict
from typing import Set, FrozenSet, Tuple, Dict

from src.main import Checker

CHARSET = string.digits[:3]
MAX_LENGTH = 3
MAX_TURNS = 5


def dfs(current: int, availables: Set[str], checker: Checker,  full_set: FrozenSet[str]) -> Tuple[dict, int]:
    if current > MAX_TURNS:
        return {}, -1

    best_guess = None
    best_guess_count = -1
    best_decision_tree = None

    # In 5 charset and length 2, full_set get 3.0 while availables get 3.08
    for guess in sorted(full_set):
    # for guess in availables:

        # 根据 pattern 对 available 分组
        # TODO: 计算完pattern之后，可以计算所有guess的期望信息量大小，优先搜索期望信息量大的guess
        pattern_results = defaultdict(set)
        for target in sorted(availables):
            pattern = checker.check(target, guess)
            pattern_results[pattern].add(target)

        # 如果当前guess不能有效缩小available size，跳过
        if len(pattern_results.keys()) == 1 and not checker.is_success(list(pattern_results.keys())[0]):
            max_pattern_count = 0
            for pattern, pattern_available in pattern_results.items():
                max_pattern_count = max(max_pattern_count, len(pattern_available))
            if max_pattern_count == len(availables):
                continue
        
        # guess_count_sum 是对于当前这个 guess，剩下所有可能结果的猜测次数之和, 而每个结果都至少还要猜1次，后面的post_guess_count是不含本次猜测结果的。
        guess_count_sum = len(availables)
        decision_tree = {}

        for pattern, pattern_availables in pattern_results.items():
            if checker.is_success(pattern):
                post_decision_tree: Dict[str, dict] = {}
                post_guess_count = 0
            else:
                # 两个返回值，best_guesses 是当前状态的最优决策树，guess_count 是对于这棵决策树来说 pattern_availables 里每个可行解的猜测次数
                (post_decision_tree, post_guess_count) = dfs(
                    current + 1,
                    pattern_availables,
                    checker,
                    full_set
                )

                if post_decision_tree == None:
                    break
            
            decision_tree[pattern] = post_decision_tree
            guess_count_sum += post_guess_count

            if best_guess_count != -1 and (guess_count_sum > best_guess_count): 
                break 
        
        # 进入 else 分支说明所有pattern都是valid
        else: 
            # 如果当前 guess 优于历史值，更新记录
            if best_guess_count == -1 or guess_count_sum < best_guess_count: 
                best_guess_count = guess_count_sum
                best_guess = guess
                best_decision_tree = decision_tree
    

    if best_guess is None:
        return {}, -1
    
    return {best_guess: best_decision_tree}, best_guess_count


def main():
    checker = Checker(MAX_LENGTH)

    # full_set = frozenset("".join(c) for c in product(CHARSET, repeat=MAX_LENGTH))
    full_set = frozenset({'233', '232', '231'})

    assert all(len(x) == MAX_LENGTH for x in full_set)

    decision_tree, post_guess_count = dfs(0, set(full_set), checker, full_set)

    print(decision_tree)
    print(post_guess_count / len(full_set))

if __name__ == '__main__':
    main()
