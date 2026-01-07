import math


def n_grams_dict(n, sentence_str: str):
    tokens = sentence_str.split()
    grams = {}
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i + n])
        grams[g] = grams.get(g, 0) + 1
    return grams

def n_gram_precision(reference_dict_list, candidate_dict, eps=1e-9):
    numerator = 0
    denominator = 0
    for g, c_cnt in candidate_dict.items():
        max_ref = 0
        for ref_dict in reference_dict_list:
            max_ref = max(max_ref, ref_dict.get(g, 0))
        numerator += min(c_cnt, max_ref)
        denominator += c_cnt

    if denominator == 0:
        return eps  # candidate 太短（没有 n-gram），避免除0
    p = numerator / denominator
    return max(p, eps)  # 避免 log(0)

def closest_reference_length(reference_list, c_len: int) -> int:
    # 选 |len(r)-len(c)| 最小的 reference，若平局取更短的
    ref_lens = [len(r.split()) for r in reference_list]
    best = ref_lens[0]
    best_diff = abs(best - c_len)
    for rl in ref_lens[1:]:
        diff = abs(rl - c_len)
        if diff < best_diff or (diff == best_diff and rl < best):
            best = rl
            best_diff = diff
    return best
def brevity_penalty(reference_list, candidate_str: str):
    c_len = len(candidate_str.split())
    if c_len == 0:
        return 0.0
    r_len = closest_reference_length(reference_list, c_len)
    if c_len > r_len:
        return 1.0
    return math.exp(1 - r_len / c_len)

def bleu(reference_list, candidate_str, ngrams=(1, 2), weights=None):
    if weights is None:
        weights = [1.0 / len(ngrams)] * len(ngrams)

    value = 0.0
    for w, n in zip(weights, ngrams):
        c_dict = n_grams_dict(n, candidate_str)
        r_dict_list = [n_grams_dict(n, r) for r in reference_list]
        p = n_gram_precision(r_dict_list, c_dict)
        value += w * math.log(p)

    bp = brevity_penalty(reference_list, candidate_str)
    return bp * math.exp(value)


r1 = "resources have to be sufficient and they have to be predictable"
r2 = "adequate and predictable resources are required"

c1 = "there is a need for adequate and predictable resources"
c2 = "resources be sufficient and predictable to"

print("BLEU(c1) =", round(bleu([r1, r2], c1, ngrams=(1,2)), 3))
print("BLEU(c2) =", round(bleu([r1, r2], c2, ngrams=(1,2)), 3))
