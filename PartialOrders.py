# Check empirically permutations inequality
import random
import numpy as np
import math
import copy


import itertools

# Find a random partition of the set ls into k non-empty sets
# From stackoverflow here: https://stackoverflow.com/questions/45829748/python-finding-random-k-subset-partition-for-a-given-list
def random_ksubset(ls, k):
    # we need to know the length of ls, so convert it into a list
    ls = list(ls)
    # sanity check
    if k < 1 or k > len(ls):
        return []
    # Create a list of length ls, where each element is the index of
    # the subset that the corresponding member of ls will be assigned
    # to.
    #
    # We require that this list contains k different values, so we
    # start by adding each possible different value.
    indices = list(range(k))
    # now we add random values from range(k) to indices to fill it up
    # to the length of ls
    indices.extend([random.choice(list(range(k))) for _ in range(len(ls) - k)])
    # shuffle the indices into a random order
    random.shuffle(indices)
    # construct and return the random subset: sort the elements by
    # which subset they will be assigned to, and group them into sets
    return [{x[1] for x in xs} for (_, xs) in
            itertools.groupby(sorted(zip(indices, ls)), lambda x: x[0])]


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
# Generate all permutations of length n with an iterator
def permutations_iter(n):
    if n == 1:
        yield [0]
    else:
        for p in permutations_iter(n-1):
            for i in range(n):
                yield p[:i] + [n-1] + p[i:]


# Generate all permutations consistent with an edge set e
# Input:
# n - number of elements
# e - list of edges (i,j) such that we must obey pi[i] < pi[j] in the permutation
#
# Output:
# iterator yielding permutations satisfying pi[i] < pi[j] for ALL edges (i,j)
def partial_order_extension(n, e):
#    print('Start iterator!')
    if n == 1:
        yield [0]
    else:
        f = [edge for edge in e if n-1 == edge[0]]  # (n-1, j)
        g = [edge for edge in e if n-1 == edge[1]]  # (j, n-1)
        for p in partial_order_extension(n-1,  [edge for edge in e if n-1 not in edge]):  # take away one element
            for i in range(n):  # where to insert n-th element
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


# Generate all permutations consistent with AT LEAST ONE edge in a set
# Input:
# n - number of elements
# e - list of edges (i,j) such that we must obey pi[i] < pi[j] for at least one edge in the permutation
#
# Output:
# iterator yielding permutations satisfying pi[i] < pi[j] for ANY of the edges (i,j)
# Runs in ~ 1 sec. or less for n <= 10
def any_pair_order_extension(n, e):
    for p in permutations_iter(n):  # just loop over all permutations and check at the end
        for edge in e:
            if p.index(edge[0]) < p.index(edge[1]):  # at least one is good
                yield p
                break


# check if a permutation p satisfies ALL pairwise orders given in F
# Input:
# p - a permutation
# F - list of lists with edges
# Output:
# Boolean: INTERSECTION of INTERSECTIONS
def check_perm_pairs_order_intersect_of_intersects(p, F):
    for E in F:
        if any([p.index(edge[1]) < p.index(edge[0]) for edge in E]):
            return False
    return True


# check if a permutation p satisfies AT LEAST one from a SET of pairwise orders given in F
# Input:
# p - a permutation
# F - list of lists with edges
# Output:
# Boolean: UNION of INTERSECTIONS
def check_perm_pairs_order_union_of_intersects(p, F):
    for E in F:
        if all([p.index(edge[1]) > p.index(edge[0]) for edge in E]):
            return True
    return False


def check_perm_pairs_order_union_of_unions(p, F):
    for E in F:
        if any([p.index(edge[1]) > p.index(edge[0]) for edge in E]):
            return True
    return False


def check_perm_pairs_order_intersect_of_unions(p, F):
    for E in F:
        if all([p.index(edge[1]) < p.index(edge[0]) for edge in E]):
            return False
    return True


# Input:
# n - number of elements
# F - a list of lists of edges
# Output:
# number of permutations that are consistent with all of the lists in F  (INTERSECT of INTERSECTIONS)
def count_intersect_posets(n, F, print_perms = False):
    ctr = 0
    for p in partial_order_extension(n,  list(set(sum(F, [])))):
        ctr += 1
        if print_perms:
            print(p)
    return ctr


# Input:
# n - number of elements
# F - a list of lists of edges
# Output:
# set of permutations that are consistent with AT LEAST one element of ALL of the lists in F  (INTERSECTIONS of UNIONS )
def get_intersect_of_unions_posets(n, F):
    return {tuple(p) for p in any_pair_order_extension(n, F[0]) if check_perm_pairs_order_intersect_of_unions(p, F)}


def count_intersect_of_unions_posets(n, F, print_perms = False):
    ctr = 0
    for p in any_pair_order_extension(n, F[0]):  # generate to match the first
        good_flag = True
        good_flag = check_perm_pairs_order_intersect_of_unions(p, F)
#        for i in range(1, len(F)):  # loop on the rest
#            if not check_perm_pairs_order_union_of_unions(p, [F[i]]):  # doesn't work
#                good_flag = False
#                break
        if print_perms and good_flag:
            print(p)
        ctr += good_flag

    return ctr


# Input:
# n - number of elements
# E - a list of lists of edges
# Output:
# set of permutations that are consistent with AT LEAST one of the lists in E  (UNION of INTERSECTIONS)
def get_union_of_intersects_posets(n, F):
    perms = {tuple(p) for p in partial_order_extension(n, F[0])}  # generate to match the first
    for i in range(1, len(F)):  # loop on the rest
        perms = perms.union({tuple(p) for p in partial_order_extension(n, F[i])})
    return perms


# Input:
# n - number of elements
# E - a list of lists of edges
# Output:
# number of permutations that are consistent with at least one of the lists in E  (Union of intersections)
def count_union_of_intersects_posets(n, F, print_perms = False):
    if print_perms:
        for p in get_union_of_intersects_posets(n, F):
            print(p)
    return len(get_union_of_intersects_posets(n, F))


# Check if the correlation inequality holds for
# n - number of variables
# F - a set of sets of edges for the conditioning part
# G - a set of sets of edges for conditioning on
def check_DNF_CNF_inequality(n, F, G):
    denom1 = count_union_of_intersects_posets(n, G)
    denom2 = count_intersect_posets(n, G)
#    num1 = count_union_of_intersects_posets(n, F + G)  # union of intersects

    U = get_union_of_intersects_posets(n, F)  # union of intersection
    V = get_union_of_intersects_posets(n, G) # union of intersection
    num1 = len(U.intersection(V))  # Compute num1
    num2 = 0  # Compute num2
    for p in partial_order_extension(n,  list(set(sum(G, [])))):
        if tuple(p) in U:
            num2 += 1

    print(num1, "/", denom1, " = ", round(num1/denom1, 3), " < ", round(num2/denom2, 3),  " = ", num2, "/", denom2)
    return num1 * denom2 <= num2 * denom1  # check inequality


# Check if the correlation inequality holds for FLIPPED ORDER (Intersection of unions!)
# n - number of variables
# F - a set of sets of edges for the conditioning part
# G - a set of sets of edges for conditioning on
def check_CNF_DNF_inequality(n, F, G):
    denom1 = count_intersect_of_unions_posets(n, G)
    denom2 = count_intersect_posets(n, G)
    #    num1 = count_union_of_intersects_posets(n, F + G)  # union of intersects

    U = get_intersect_of_unions_posets(n, F)  # union of intersection. NEED TO IMPLEMENT!
    V = get_intersect_of_unions_posets(n, G) # union of intersection
    num1 = len(U.intersection(V))  # Compute num1
    num2 = 0  # Compute num2
    for p in partial_order_extension(n,  list(set(sum(G, [])))):  # CHANGE HERE TOO!!!
        if tuple(p) in U:
            num2 += 1

    print(num1, "/", denom1, " = ", round(num1/denom1, 3), " < ", round(num2/denom2, 3),  " = ", num2, "/", denom2)
    return num1 * denom2 <= num2 * denom1  # check inequality


# New! check the special case with rows and columns (Theorem 2 in the paper)
# Convension: X00, ..., X0(k-1), X10,..,X1(k-1),...,x(n-1)0,...,x(n-1)(k-1)
def check_matrix_CNF_DNF_inequality(n_rows, k_cols):
    n = n_rows * k_cols

    # generate F and G
    F = [[]]*(n_rows-2)
    G = [[]]*(n_rows-2)
    for i in range(2, n_rows):  # go from 3 to ...
        print("i=", i)
        F[i-2] = [(i*k_cols + k, 1*k_cols + k) for k in range(k_cols)]
        G[i-2] = [(i*k_cols + k, k) for k in range(k_cols)]

    print("F=", F, "\nG=", G)
    return check_CNF_DNF_inequality(n, F, G)


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


# Randomize triplets of vertices/random-variables (i, j, k)
# Input:
# n - total size
# B_1 - set of first vertex
# B_2 - set of second vertex
# m - number of edges in each subset
# T - number of subsets
# unique_type - (default: False). If true, then for each of the T subsets, we give a DIFFERENT variable for the inequality
#               fix - (default: False). If true, then we allow only triplets with the same pair.
#               That is, we can't have (i, j1, k1) and (i, j2, k2)
#               "edge|: If true, then every edge (i,j) where i in B_1 and j in B_2 can appear only in one of the T claueses,
#               and but we allow different k's to be shared
#               "None" - allow any triplets(i,j,k) with i <- B_1, j <- B_2, k <- C
#               "diff_k" - allow i <- B_1, j <- B_2, k <- C_t   for clause t=1,..,T
# Output:
# A list F of lists. Each list F[l] is a list of triplets (i, j, k), such that we have the following probabilities:
# Pr( \intersect_l \union_{(i, j, k) \in F[l]} {X_i > X_k} | \intersect_l \union_{(i, j, k) \in F[l]} {X_j > X_k})
# on the L.H.S. of the inequality, and where the union is replaced by another intersection in the R.H.S.
def gen_random_triplets_partial_orders(n, B_1, B_2, m, T, unique_type = False):
    B = list(set(B_1).union(B_2))  # B_1 union B_2
    C = [i for i in range(n) if i not in B]  # Complement, [1:n] \ B = B_c

    # Replacement for switch
    if unique_type == "vertex":  # here partition the rest n elements into T disjoint subsets
        C_vec = random_ksubset(C, T)
    if unique_type == "edge":  # here do not allow the same variables from B_1 or B_2 in different sets
        E = [(i, j) for i in B_1 for j in B_2]
#        print("B1 B2 Size:", len(B_1), len(B_2))
#        print("E Size, m:", len(E), m)

        USE_E = random.sample(E, m)

    if unique_type == "edge_pairs":  # here set (b_11, b_21), .., (b_1m, b_2m)
        USE_E = [(i, m+i) for i in range(m)] # B_1 = (0,..,m-1), B_2 = (m, .., 2m-1)

    F = [[0]]*T
    for t in range(T):  # sample triplets
        if unique_type == "vertex":  # Different groups have different C vertices
            F[t] = [(random.sample(B_1, 1)[0], random.sample(B_2, 1)[0], random.sample(C_vec[t], 1)[0]) for j in range(m)]
        if unique_type in ["edge", "edge_pairs"]:   # need m edges for (i,j) !! They go together repeatedly with different k values !!
            F[t] = [(USE_E[j][0], USE_E[j][1], random.sample(C, 1)[0]) for j in range(m)]
        if unique_type == False:   # non unique, allow any triplets (i,j,k)
            F[t] = [(random.sample(B_1, 1)[0], random.sample(B_2, 1)[0], random.sample(C, 1)[0]) for j in range(m)]


    return F


# Draw random subsets of edges and try to find a violation of the inequality
# DNF_CNF_order - UNION of INTERSECTIONS (default, True), or INTERSECTION of UNIONS (False), as in Theorem
# unique_type - If "vertex", we must have different vertices. If "edges", we must have triplets with the same (i,j)
# If false, no restriction on the
# Input:
# n - total number of random variables
# B_1 - first set (numerator)
# B_2 - second set (denominator)
# num_edges - how many pairwise constraint in each intersection/union event set
# num_sets - how many sets of intersection/union
# iters - how many random sets to draw
# full_print - print entire permutations or just their numbers
# triplets - True, do the (i,j,k) in triplets, meaning that for each i in B_1 and j in B_2 the same k is applied
# unique_type - different types of constraints on the events:
#             (*) False - not constrained
#             (*) "vertex" - each (i, j, k) the k's for different sets must be different
#             (*) "edge" - Each (i,j) appears only once and in every constraint
def find_random_counter_example_DNF_CNF_inequality(n = 4, B_1 = [0, 1], B_2 = [],
                                                   num_edges = 3, num_sets = 2, iters = 100,
                                                   full_print = False, triplets = False,
                                                   unique_type = False, DNF_CNF_order = True):

    #####################################################
    #### Set parameters not fiven/given partially #######
    if len(num_edges) == 1:
        num_edges_vec = [num_edges]*num_sets
    if len(B_2) == 0:  # not given
        B_2 = B_1  # no need for deep copy (we don't alter B)
    #####################################################


    B = list(set(B_1).union(B_2))

    # Check inequality
    #    n = 4
    #    B = [0, 1] # ,3]
    B_c = [i for i in range(n) if i not in B]  # we call it C now
    m = num_edges  # 3  # number of edges in set
    T = num_sets  # 2  # number of sets
    #    iters = 100

    for i in range(iters):
        if triplets:  # NEW!!!
            F3 = gen_random_triplets_partial_orders(n, B_1, B_2, m, T, unique_type)

            F = [[(x[0], x[2]) for x in y] for y in F3]  # take pairs out of the triplet
            G = [[(x[1], x[2]) for x in y] for y in F3]
        else:
            F = gen_random_partial_orders(n, B_1, m, T, B_c)  # change the conditioning and conditioned sets
            G = gen_random_partial_orders(n, B_2, m, T, B_c)

        if DNF_CNF_order:
            check = check_DNF_CNF_inequality(n, F, G)
        else:
            check = check_CNF_DNF_inequality(n, F, G)
        print("iter:", i, "n=", n, "T=", num_sets, "m=", num_edges, "check:", check)
#        if (DNF_CNF_order and not check_DNF_CNF_inequality(n, F, G)) or (not DNF_CNF_order and not check_CNF_DNF_inequality(n, F, G)):
        if not check:
            print("Error! Inequality Doesn't Hold!")
            print("F:", F)
            print("G:", G)
            print("Error! Inequality Doesn't Hold! n=" + str(n), ", B_1 = ", B_1, ", B_2 = ", B_2, " ; B_c=", B_c)

            if full_print:
                if DNF_CNF_order:  # unions of intersects, just for printing!!!
                    denom1_bad_perms = [p for p in permutations_iter(n) if check_perm_pairs_order_union_of_intersects(p, G)]
                    num1_bad_perms = [p for p in denom1_bad_perms if check_perm_pairs_order_union_of_intersects(p, F)]
                    denom2_bad_perms = [p for p in partial_order_extension(n, list(set(sum(G, []))))]  # intersect of intersect
                    num2_bad_perms = [p for p in denom2_bad_perms if check_perm_pairs_order_union_of_intersects(p, F)]
                else:  # INTERSECTS of UNIONs, just for printing!!!
                    denom1_bad_perms = [p for p in permutations_iter(n) if check_perm_pairs_order_intersect_of_unions(p, G)]
                    num1_bad_perms = [p for p in denom1_bad_perms if check_perm_pairs_order_intersect_of_unions(p, F)]
                    denom2_bad_perms = [p for p in partial_order_extension(n, list(set(sum(G, []))))]  # ???  intersect of intersect
                    num2_bad_perms = [p for p in denom2_bad_perms if check_perm_pairs_order_intersect_of_unions(p, F)]
            else:  # do not compute permutations
                num1_bad_perms = num2_bad_perms = denom1_bad_perms = denom2_bad_perms = []

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
                print("Num bad perms:", len(num1_bad_perms), len(num2_bad_perms), len(denom1_bad_perms), len(denom2_bad_perms), )
            return F, G, denom1_bad_perms, denom2_bad_perms, num1_bad_perms, num2_bad_perms

    print("Didn't find any counter-example after ", iters, " iters!!!")
    return False
