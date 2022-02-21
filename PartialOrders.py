# Check empirically permutations inequality
import random
import numpy as np
import math
import copy




# Permutations function:
def permutations(lst):
    if len(lst) == 1:
        return [lst]
    else:
        perm = []
    for i in range(len(lst)):
        perm = perm + [[lst[i]]+l for l in
            permutations(lst[:i]+lst[i+1:])]
    return perm


# Permutations iterator:
def permutations_iter(n):
    if n == 1:
        yield [0]
    else:
        for p in permutations_iter(n-1):
            for i in range(n):
                yield p[:i] + [n-1] + p[i:]


# Generate all permutations consistent with an edge set e
def partial_order_extension(n, e):
#    print('Start iterator!')
    if n == 1:
        yield [0]
    else:
        f = [edge for edge in e if n-1 == edge[0]]  # (n-1 , j)
        g = [edge for edge in e if n-1 == edge[1]]  # (j, n-1)
        for p in partial_order_extension(n-1,  [edge for edge in e if n-1 not in edge]): # take away one element
            for i in range(n):
                good_perm = True
                for edge in f:  # check consistency with pairs
                    if edge[1] in p[:i]:
                        good_perm = False
                        break
                if good_perm:
                    for edge in g:  # check consistency with pairs
                        if edge[0] in p[i:]:
                            good_perm = False
                            break
                    if good_perm:
                        yield p[:i] + [n-1] + p[i:]


# check if a permutation p satisfies a set of pairwise orders given in F
def check_perm_pairs_order_intersect(p, F):
    for E in F:
        if any([p.index(edge[1]) < p.index(edge[0]) for edge in E]):
            return False
    return True


# check if a permutation p satisfies at least one from a set of pairwise orders given in F
def check_perm_pairs_order_union(p, F):
    for E in F:
        if all([p.index(edge[1]) > p.index(edge[0]) for edge in E]):
            return True
    return False


# Input:
# n - number of elements
# E - a list of lists of edges
# Output:
# number of permutations that are consistent with all of the lists in E  (Union of intersections)
def count_intersect_posets(n, F):
    ctr = 0
    for p in partial_order_extension(n,  list(set(sum(F, [])))):
        ctr += 1
    return ctr


# Input:
# n - number of elements
# E - a list of lists of edges
# Output:
# set of permutations that are consistent with at least one of the lists in E  (Union of intersections)
def get_union_posets(n, F):
    perms = {tuple(p) for p in partial_order_extension(n, F[0])}
    for i in range(1, len(F)):  # loop on the rest
        perms = perms.union({tuple(p) for p in partial_order_extension(n, F[i])})
    return perms


# Input:
# n - number of elements
# E - a list of lists of edges
# Output:
# number of permutations that are consistent with at least one of the lists in E  (Union of intersections)
def count_union_posets(n, F):
    return len(get_union_posets(n, F))


# Check if the correlation inequality holds for
# n - number of variables
# F - a set of sets of edges for the conditioning part
# G - a set of sets of edges for conditioning on
def check_DNF_CNF_inequality(n, F, G):
    denom1 = count_union_posets(n, G)
    denom2 = count_intersect_posets(n, G)
#    num1 = count_union_posets(n, F + G)  # union of intersects

    U = get_union_posets(n, F)
    V = get_union_posets(n, G)
    num1 = len(U.intersection(V))  # Compute num1
    num2 = 0  # Compute num2
#    print(U)
    for p in partial_order_extension(n,  list(set(sum(G, [])))):
#        print(p)
        if tuple(p) in U:
            num2 += 1
#        num2 += p in U

    print(num1, denom1, num2, denom2)
    return num1 * denom2 <= num2 * denom1  # check inequality


# Randomize edges
# Input:
# n - total size
# B - set of first vertex
# m - number of edges
# T - number of subsets
def gen_random_partial_orders(n, B, m, T, C=[]):
    if len(C) == 0:  # default
        C = [i for i in range(n) if i not in B]  # Complement, B_c
    F = [[0]]*T
    for t in range(T):
        F[t] = [(random.sample(B, 1)[0], random.sample(C, 1)[0]) for j in range(m)]
    return F





# Draw random subsets of edges and try to find a violation of the inequality
def find_random_counter_example_DNF_CNF_inequality(n = 4, B_1 = [0, 1], B_2 = [], num_edges = 3, num_sets = 2, iters = 100, full_print = False):
    if len(B_2) == 0: # not given
        B_2 = B_1  # no need for deep copy (we don't alter B)
    B = list(set(B_1).union(B_2))

    # Check inequality
    #    n = 4
    #    B = [0, 1] # ,3]
    B_c = [i for i in range(n) if i not in B]  # we call it C now
    m = num_edges # 3  # number of edges in set
    T = num_sets # 2  # number of sets
    #    iters = 100

    for i in range(iters):
        F = gen_random_partial_orders(n, B_1, m, T, B_c)  # change the conditioning and conditioned sets
        G = gen_random_partial_orders(n, B_2, m, T, B_c)
        print(check_DNF_CNF_inequality(n, F, G))
        if not check_DNF_CNF_inequality(n, F, G):
            print("Error! Inequality Doesn't Hold!")
            print("F:", F)
            print("G:", G)
            print("Error! Inequality Doesn't Hold! n=" + str(n), ", B_1 = ", B_1, ", B_2 = ", B_2, " ; B_c=", B_c)

            denom1_bad_perms = [p for p in permutations_iter(n) if check_perm_pairs_order_union(p, G)]
            num1_bad_perms = [p for p in denom1_bad_perms if check_perm_pairs_order_union(p, F)]
            denom2_bad_perms = [p for p in partial_order_extension(n, list(set(sum(G, []))))]
            num2_bad_perms = [p for p in denom2_bad_perms if check_perm_pairs_order_union(p, F)]

            if full_print:
                print("Bad permutations for denom1:")
                for p in denom1_bad_perms:
                    print([x+1 for x in p])
                print("Bad permutations for num1:")
                for p in num1_bad_perms:
                    print([x+1 for x in p])
                print("Bad permutations for denom2:")
                for p in denom2_bad_perms:
                    print([x+1 for x in p])
                print("Bad permutations for num2:")
                for p in num2_bad_perms:
                    print([x+1 for x in p])

            return F, G, denom1_bad_perms, denom2_bad_perms, num1_bad_perms, num2_bad_perms
#            break

    print("Didn't find any counter-example after ", iters, " iters!!!")
    return False
