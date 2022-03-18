import string
from typing import Tuple, List, Dict, Set

def uniform(availables: List[str]) -> Tuple[str, str]:
    # 统计字符
    char_set = sorted({c for word in availables for c in word })
    # 转换成最小字符
    mapping = {}
    backward = ''
    for idx, c in enumerate(char_set):
        mapping[c] = chr(idx + 97)
        backward += c

    key = ''.join([mapping[c] for word in availables for c in word])

    return key, backward

def decode(decision_tree: Dict[str, dict], backward: str) -> Dict[str, dict]:
    backward_mapping = {chr(idx + 97): c  for idx, c in enumerate(backward)}

    new_tree = {}
    for k, v in decision_tree.items():
        if all([c in string.ascii_lowercase for c in k]):
            new_key = ''.join([backward_mapping[c] for c in k])
            new_tree[new_key] = decode(v, backward)
        else:
            new_tree[k] = decode(v, backward)
    
    return new_tree

def encode(decision_tree: Dict[str, dict], backward: str) -> Dict[str, dict]:
    forward_mapping = {c: chr(idx + 97)  for idx, c in enumerate(backward)}
    new_tree = {}
    for k, v in decision_tree.items():
        if all([c in string.ascii_lowercase for c in k]):
            new_key = ''.join([forward_mapping[c] for c in k])
            new_tree[new_key] = encode(v, backward)
        else:
            new_tree[k] = encode(v, backward)
    
    return new_tree

def encode_tree(decision_tree: Dict[str, dict], origin_backward: str) -> Tuple[Dict[str, dict], str]:

    def get_tree_node_char_set(decision_tree) -> Set[str]:
        char_set: Set[str] = set()
        for k, v in decision_tree.items():
            if all([c in string.ascii_lowercase for c in k]):
                for c in k:
                    char_set.add(c)
            for c in get_tree_node_char_set(v):
                char_set.add(c)
            
        return char_set

    for c in sorted(get_tree_node_char_set(decision_tree)):
        if c not in origin_backward:
            origin_backward += c
    
    mapping = {}
    backward = ''
    for idx, c in enumerate(origin_backward):
        mapping[c] = chr(idx + 97)
        backward += c
    
    return (encode(decision_tree, backward), backward)

# abcdef
# adfhrw

# abcdefghi
# abchknquw

def main():
    # print(uniform(['dwarf', 'wharf']))
    print(encode({'dwarf': {'GGGGG': {}, 'BYGGG': {'wharf': {'GGGGG': {}}}}}, 'adfhrw'))
    print(decode({'bfaec': {'GGGGG': {}, 'BYGGG': {'fdaec': {'GGGGG': {}}}}}, 'adfhrw'))

if __name__ == '__main__':
    main()
