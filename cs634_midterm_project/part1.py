import sys
import time

min_support = float(sys.argv[1])
min_conf = float(sys.argv[2])
t_file = sys.argv[3]

with open(t_file) as f:
    db = [l.replace("\n", "").split(",") for l in f]
    print("------------------------------------------------------------ INPUT TRANSACTIONS:")
    for transaction in db:
        print(transaction)


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


def scan(db, Ck):
    count = {s: 0 for s in Ck}
    for t in db:
        for freqset in Ck:
            if freqset.issubset(t):
                count[freqset] += 1
    n = len(db)
    return {freqset: support/n for freqset, support in count.items() if support/n>=min_support}


def generate_candidate(Lk):
    result = []
    for i in range(len(Lk)):
        for j in range(i+1, len(Lk)):
            a, b = Lk[i], Lk[j]
            aa, bb = list(a), list(b)
            aa.sort()
            bb.sort()
            if aa[:len(a)-1] == bb[:len(a)-1]:
                result.append(a | b)
    return result


def generate_frequent_and_support():
    support = {}
    candidate = [[]]
    Lk = [[]]
    C1 = set()
    for t in db:
        for item in t:
            C1.add(frozenset([item]))

    candidate.append(C1)
    count = scan(db, C1)
    Lk.append(list(count.keys()))
    support.update(count)

    k = 1
    while len(Lk[k]) > 0:
        candidate.append(generate_candidate(Lk[k]))
        count = scan(db, candidate[k+1])
        support.update(count)
        Lk.append(list(count.keys()))
        k += 1
    return Lk, support


def generate_sub_rule(fs, rights, all_result, support):
    right_size = len(rights[0])
    total_size = len(fs)
    if total_size-right_size > 0:
        rights = generate_candidate(rights)
        new_right = []
        for right in rights:
            left = fs - right
            if len(left) == 0:
                continue
            confidence = support[fs] / support[left]
            if confidence >= min_conf:
                all_result.append([Rule(left, right, fs), support[fs],  confidence])
                new_right.append(right)

        if len(new_right) > 1:
            generate_sub_rule(fs, new_right, all_result, support)


def generate_rules(frequent, support):
    all_result = []
    for i in range(2, len(frequent)):
        if len(frequent[i]) == 0:
            break
        freq_sets = frequent[i]

        for fs in freq_sets:
            for right in [frozenset([x]) for x in fs]:
                left = fs-right
                confidence = support[fs] / support[left]
                if confidence >= min_conf:
                    all_result.append([Rule(left, right, fs), support[fs], confidence])

        if len(freq_sets[0]) != 2:

            for fs in freq_sets:
                right = [frozenset([x]) for x in fs]
                generate_sub_rule(fs, right, all_result, support)

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
