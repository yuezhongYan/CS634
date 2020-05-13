import sys
import time

min_support = float(sys.argv[1])
min_conf = float(sys.argv[2])
t_file = sys.argv[3]

with open("dataset/item.csv") as f:
    items = f.read().replace("\n", "").split(",")
    items.sort()

with open(t_file) as f:
    db = [l.replace("\n", "").split(",") for l in f]
    print("------------------------------------------------------------ INPUT TRANSACTIONS:")
    for transaction in db:
        print(transaction)


def generate_k(items, k):

    if k == 1:
        return [[x] for x in items]

    all_res = []
    for i in range(len(items)-(k-1)):
        for sub in generate_k(items[i+1:], k-1):
            tmp = [items[i]]
            tmp.extend(sub)
            all_res.append(tmp)
    return all_res


def scan(db, s):
    count = 0
    for t in db:
        if set(s).issubset(t):
            count += 1
    return count


def generate_frequent_and_support():
    frequent = []
    support = {}
    for k in range(1, len(items)+1):
        current = []
        for comb in generate_k(items, k):
            count = scan(db, comb)
            if count/len(db) >= min_support:
                support[frozenset(comb)] = count/len(db)
                current.append(comb)
        if len(current) == 0:
            break
        frequent.append(current)
    return frequent, support


class Rule:

    def __init__(self, left, right, all):
        self.left = list(left)
        self.left.sort()
        self.right = list(right)
        self.right.sort()
        self.all = all

    def __str__(self):
        return ",".join(self.left)+" => "+",".join(self.right)

    def __hash__(self):
        """
        Store support value to dict
        :return: hash value in the object
        """
        return hash(str(self))


def generate_rules(frequent, support):
    all_rule = set()
    all_result = []
    for k_freq in frequent:
        if len(k_freq) == 0:
            continue
        if len(k_freq[0]) < 2:
            continue
        for freq in k_freq:
            for i in range(1, len(freq)):
                for left in generate_k(freq, i):
                    tmp = freq.copy()
                    right = [x for x in tmp if x not in left]
                    all_rule.add(Rule(left, right, freq))
    for rule in all_rule:
        confidence = support[frozenset(rule.all)] / support[frozenset(rule.left)]
        if confidence >= min_conf:
            all_result.append([rule, support[frozenset(rule.all)], confidence])

    all_result.sort(key=lambda x: str(x[0]))

    return all_result


if __name__ == '__main__':
    start_time = time.time()
    f, s = generate_frequent_and_support()
    all_result = generate_rules(f, s)
    end_time = time.time()
    print("\n----------------------------------------------------- RULES SUPPORT CONFIDENCE:")
    for r in all_result:
        print(r[0], r[1], r[2])
    print("\n----------------------------------------------------------------- RUNNING TIME:")
    print(str(end_time - start_time) + "s")
