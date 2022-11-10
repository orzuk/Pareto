# Check empirically permutations inequality
import random
import itertools
import numpy as np
import math
import copy
import time
from pareto import *

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


# Return all permutations of list elements:
def permutations(lst):
    if len(lst) == 1:
        return [lst]
    else:
        perm = []
    for i in range(len(lst)):
        perm = perm + [[lst[i]]+l for l in permutations(lst[:i]+lst[i+1:])]
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
def permutations_partial_order_iter(n, e):
    if n == 1:
        yield [0]
    else:
        f = [edge for edge in e if n-1 == edge[0]]  # (n-1, j)
        g = [edge for edge in e if n-1 == edge[1]]  # (j, n-1)
        for p in permutations_partial_order_iter(n-1,  [edge for edge in e if n-1 not in edge]):  # take away one element
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
def permutations_any_pair_order_iter(n, e):
    for p in permutations_iter(n):  # just loop over all permutations and check at the end. Could be costly , but can't save much (union has > half of permutations)
        for edge in e:
            if p.index(edge[0]) < p.index(edge[1]):  # at least one is good
                yield p
                break  # break inner loop, continue to next permutation


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


# check if a permutation p satisfies AT LEAST one from a SET of pairwise orders given in F, ALL orders in a set
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


# check if a permutation p satisfies AT LEAST one from a SET of pairwise orders given in F, AT LEAST one order in a set
def check_perm_pairs_order_union_of_unions(p, F):
    for E in F:
        if any([p.index(edge[1]) > p.index(edge[0]) for edge in E]):
            return True
    return False


# check if a permutation p satisfies ALL SETS of pairwise orders given in F, AT LEAST one order in a set
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
    new_n, new_F = reduce_posets(n, F)  # take only relevant variables that appear in F
    if new_n < n and not print_perms:  # account for dropped variables
        return count_intersect_posets(new_n, new_F) * math.prod(range(new_n+1, n+1))

    ctr = 0
    for p in permutations_partial_order_iter(n,  list(set(sum(F, [])))):
        ctr += 1
        if print_perms:
            print(p)
    return ctr


# Use recursive formula
def count_intersect_posets2(n, F_flat):
    if len(F_flat)==0:
        return math.factorial(n)  # all possible orders
    maximal = set(range(n)) - {x[0] for x in F_flat} # Get maximal
    r = 0
    for x in maximal:
        r += count_intersect_posets2(n-1, [(y[0] if y[0]<x else y[0]-1, y[1] if y[1]<x else y[1]-1) for y in F_flat if y[1] != x])
    return r


# Input:
# n - number of elements
# F - a list of lists of edges
# Output:
# set of permutations that are consistent with AT LEAST one element of ALL of the lists in F  (INTERSECTIONS of UNIONS )
def get_intersect_of_unions_posets(n, F):
    return {tuple(p) for p in permutations_any_pair_order_iter(n, F[0]) if check_perm_pairs_order_intersect_of_unions(p, F[1:])}


def count_intersect_of_unions_posets(n, F, print_perms = False):
    if not print_perms:
        return math.factorial(n) - count_union_of_intersects_posets(n, F)  # faster !! taking the complement of unions of intersections !!!

    ctr = 0
    for p in permutations_any_pair_order_iter(n, F[0]):  # generate to match the first
        if check_perm_pairs_order_intersect_of_unions(p, F[1:]):  # can skip first one to save time
            print(p)
            ctr += 1
    return ctr


# Input:
# n - number of elements
# E - a list of lists of edges
# Output:
# set of permutations that are consistent with AT LEAST one of the lists in E  (UNION of INTERSECTIONS)
def get_union_of_intersects_posets(n, F):
    perms = {tuple(p) for p in permutations_partial_order_iter(n, F[0])}  # generate to match the first
    for i in range(1, len(F)):  # loop on the rest
        perms = perms.union({tuple(p) for p in permutations_partial_order_iter(n, F[i])})
    return perms


# Remove irrelevant indices
def reduce_posets(n, F):
    use_inds = list({x for y in F for z in y for x in z})
    new_F = copy.deepcopy(F)

    if len(use_inds) < n:  # save some
        use_inds_inv = [0]*n
        for i in range(len(use_inds)):
            use_inds_inv[use_inds[i]] = i

        for i in range(len(F)):
            for j in range(len(F[i])):
                new_F[i][j] = tuple(use_inds_inv[k] for k in F[i][j])
    return len(use_inds), new_F


# Input:
# n - number of elements
# E - a list of lists of edges
# Output:
# number of permutations that are consistent with at least one of the lists in E  (Union of intersections)
def count_union_of_intersects_posets(n, F, print_perms = False):
    if print_perms:
        for p in get_union_of_intersects_posets(n, F):
            print(p)

    new_n, new_F = reduce_posets(n, F)
    if new_n < n:  # save some
        return len(get_union_of_intersects_posets(new_n, new_F)) * math.prod(range(new_n+1, n+1))
    else:
        return len(get_union_of_intersects_posets(n, F))





# Check if the correlation inequality holds for UNIONS of INTERSECTIONS!
# n - number of variables
# F - a set of sets of edges for the conditioning part
# G - a set of sets of edges we condition on, for denominator
def check_DNF_CNF_inequality(n, F, G):
    return check_CNF_DNF_inequality(n, F, G, True)

#    denom1 = count_union_of_intersects_posets(n, G)
#    denom2 = count_intersect_posets(n, G)
#
#    U = get_union_of_intersects_posets(n, F)  # union of intersection
#    V = get_union_of_intersects_posets(n, G) # union of intersection
#    num1 = len(U.intersection(V))  # Compute num1
#    num2 = 0  # Compute num2
#    for p in permutations_partial_order_iter(n,  list(set(sum(G, [])))):
#        if tuple(p) in U:
#            num2 += 1
#
#    print(num1, "/", denom1, " = ", round(num1/denom1, 3), " < ", round(num2/denom2, 3),  " = ", num2, "/", denom2)
#    return num1 * denom2 <= num2 * denom1, num1, num2, denom1, denom2   # check inequality


# Check if the correlation inequality holds for FLIPPED ORDER (INTERSECTION of UNIONS!, like inequality in the paper!)
# n - number of variables
# F - a set of sets of edges for the conditioning part
# G - a set of sets of edges for conditioning on
# DNF_CNF_order - False: CNF_DNF (default): INTERSECTION of UNIONS!. True: DNF_CNF : UNION of INTERSECTIONS!.
# iters - optional, for checking first with sampling
def check_CNF_DNF_inequality(n, F, G, DNF_CNF_order = False, iters = -1, print_flag=False):
    use_inds =  list( set([x for y in F for z in y for x in z]).union( [x for y in G for z in y for x in z]  ) )
    use_inds_inv = [0]*n
    for i in range(len(use_inds)):
        use_inds_inv[use_inds[i]] = i

    if len(use_inds) < n:  # save some, remove irrelevant variables
        new_G = copy.deepcopy(G)
        for i in range(len(G)):
            for j in range(len(G[i])):
                new_G[i][j] = tuple(use_inds_inv[k] for k in G[i][j])
        new_F = copy.deepcopy(F)
        for i in range(len(F)):
            for j in range(len(F[i])):
                new_F[i][j] = tuple(use_inds_inv[k] for k in F[i][j])

#        print("Reduce check: ", n, " -> ", len(use_inds), ". ", end="")
        return check_CNF_DNF_inequality(len(use_inds), new_F, new_G, DNF_CNF_order, iters)
    # NEW!!! Check first by sampling !!!!
    if iters > 0:
        num1, num2, denom1, denom2 = 0, 0, 0, 0
        for i in range(iters):
            P = random.sample(range(n), n)
            d1 = check_perm_pairs_order_intersect_of_unions(P, G)
            d2 = check_perm_pairs_order_intersect_of_intersects(P, G)
            denom1 += d1
            denom2 += d2
            if d1 or d2:
                n1 = check_perm_pairs_order_intersect_of_unions(P, F)
                num1 += d1*n1
                num2 += d2*n1

        sign = "<=" if num1 * denom2 <= num2 * denom1 else ">"
        if num1 * denom2 <= num2 * denom1:  # Sampling is good
            if print_flag:
                print("Good sampling:", num1, "/", denom1, " = ", round(num1/denom1, 3), sign, round(num2/denom2, 3),  " = ", num2, "/", denom2, ", ", end="")
            return True, num1, num2, denom1, denom2  # check inequality. Return boolean and four values
        else:
            if print_flag:
                print("Bad sampling:", num1, "/", denom1, " = ", round(num1/denom1, 3), sign, round(num2/denom2, 3),  " = ", num2, "/", denom2, " check exact!")

    denom2 = count_intersect_posets(n, G)  # the same for both options
    if DNF_CNF_order:  # union of intersections . Easier to count !!
        denom1 = count_union_of_intersects_posets(n, G)
        U = get_union_of_intersects_posets(n, F)  # union of intersection
        V = get_union_of_intersects_posets(n, G)  # union of intersection
    else:  # default: intersection of unions !!
        denom1 = count_intersect_of_unions_posets(n, G)
#        U = get_intersect_of_unions_posets(n, F)
#        V = get_intersect_of_unions_posets(n, G)  # intersection of union. How to enumerate all these fast??
        num1 = count_intersect_of_unions_posets(n, F + G)

    num2 = 0  # Compute numerator 2
    for p in permutations_partial_order_iter(n,  list(set(sum(G, [])))):  # CHANGE HERE TOO!!!
        if check_perm_pairs_order_intersect_of_unions(p, F):
            num2 += 1

    if print_flag:
        sign = "<=" if num1 * denom2 <= num2 * denom1 else ">"
        print(num1, "/", denom1, " = ", round(num1/denom1, 3), sign, round(num2/denom2, 3),  " = ", num2, "/", denom2)
    return num1 * denom2 <= num2 * denom1, num1, num2, denom1, denom2  # check inequality. Return boolean and four values


# New! check the special case with rows and columns (Theorem 2 in the paper)
# Convention: X_00, ..., X_0(k-1), X_10,..,X_1(k-1),...,x_(n-1)0,...,x_(n-1)(k-1)
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
# C - set of second vertex (optional)
# unique - False: randomize with replacement. True: Each edge appears only once overall!!!
# Output:
# A list of lists of edges, each list of edges representing a partial order.
def gen_random_partial_orders(n, B, m, T, C=[], unique = False):
    if not isinstance(m, list):  # all F_t, G_s have the same number of edges
        m = [m]*T
    if len(C) == 0:  # default
        C = [i for i in range(n) if i not in B]  # Complement, B_c
    F = [[0]]*T

    if unique:  # Each edge appears once overall
        USE_F = random.sample([(i, j) for i in B for j in C], sum(m))
        ctr = 0
        for t in range(T):
            F[t] = USE_F[ctr: (ctr+m[t])]
            ctr += m[t]
    else:  # here generate any edges
        for t in range(T):
            F[t] = [(random.sample(B, 1)[0], random.sample(C, 1)[0]) for j in range(m[t])]

    # New: each i appears at most once in every set F_t:
    if unique == 2:
        for t in range(T):
            source = random.sample(B, m[t])  # Source nodes
            dest = random.choices(C, k=m[t])
            F[t] = [(source[i], dest[i]) for i in range(m[t])]
    return F


# Randomize triplets of vertices/random-variables (i, j, k)
# Input:
# n - total size
# B1 - set of first vertex
# B2 - set of second vertex
# m - number of edges in each subset
# T - number of subsets
# unique_type - What constraint on the triplets do we have. (default: False).
#               False             : Sample any (i,j,k) with i<-B1, j<-B2, k<-C
#               "vertex"          : For each subset F_t sample j from a different C_t, where C=disjoint-union_t C_t
#               "vertex_distinct" : Like "vertex", but within each subset, all the values of i,j and k are different.
#               "edge"            : Every edge (i,j)<-B1*B2 can appear only once in each of the T subsets,
#                                   but we allow different k's to be shared
#               "edge_pairs"      : Like edge, but each i can have only one edge (i,j), and similarly for j. The same edge can repeat many times
#               "edge_B_C_unique"     : Do not allow the same edge (i,k) or the same edge (j,k) twice anywhere, across all clauses ... !!!!
#               "diff_k" - allow i <- B1, j <- B2, k <- C_t   for clause t=1,..,T
#               Allow only triplets with the same pair: can't have (i, j1, k1) and (i, j2, k2)
#
# Output:
# A list F of lists. Each list F[l] is a list of triplets (i, j, k), such that we have the following probabilities:
# Pr( \intersect_l \union_{(i, j, k) \in F[l]} {X_i > X_k} | \intersect_l \union_{(i, j, k) \in F[l]} {X_j > X_k})
# on the L.H.S. of the inequality, and where the union is replaced by another intersection in the R.H.S.
def gen_random_triplets_partial_orders(n, B1, B2, m, T, unique_type=False):
    if not isinstance(m, list):
        m = [m]*T  # duplicate values
    B = list(set(B1).union(B2))  # B1 union B2
    C = [i for i in range(n) if i not in B]  # Complement, [1:n] \ B = B_c

    # Replacement for switch
    if unique_type in ["vertex", "vertex_distinct"]:  # Partition the rest n elements into T disjoint subsets
        c_vec = random_ksubset(C, T)
    if unique_type == "vertex_distinct":  # Each subset t has >= m[t] elements
        if len(C) < sum(m):
            print("Error! Can't build constraints with these parameters, increase n or decrease m!!!")
            return 0
        m_cum = np.cumsum([0] + m)
        c_vec = [C[m_cum[i]:m_cum[i+1]] for i in range(T)]  # at least m[t] for set t (>= 1 in each set)
        for i in range(m_cum[T],len(C)):  # split the others
            c_vec[random.sample(range(T), 1)[0]] += [C[i]]
        c_vec = [set(s) for s in c_vec]
#        for i in range(T):
#            c_vec[i] = set(c_vec[i])

    if unique_type == "edge":  # here do not allow the same variables from B1 or B2 in different sets
        E = [(i, j) for i in B1 for j in B2]
        USE_E = random.sample(E, max(m))  # take fixed edges for all sets, but allow the same vertex to belong to multiple edges

    unique_z = False  # TWO MODES! Should be part of function flags
    if unique_type == "edge_pairs":  # here set (B11, B21), .., (B1m, B2m)
        USE_E = [(i, max(m)+i) for i in range(max(m))]  # Must have: B1 = (0,..,m-1), B2 = (m, .., 2m-1)  (ignore B1 and B2 inputs). Need to change C too !
        C = list(range(2*max(m), n))  # all others
        if unique_z and max(m)*3 > n:  # New! force all Z_i's to be distinct in the same clause
            print("Error! Can't build constraints with these parameters, increase n or decrease m!!!")
            return 0
    # new !!!
#    if unique_type == "trip_unique": # here choose ONCE the set of all edges
#        ALL_F = [(i, j, k) for i in B1 for j in B2 for k in C]
#        USE_F = random.sample(ALL_F, sum(m))  # one different triplet for each clause

    F = [[0]]*T
    for t in range(T):  # Sample triplets for every subset of edges F_t
        if unique_type == "vertex":  # Different groups have different C vertices. Distinct within each constraint
            F[t] = [(random.sample(B1, 1)[0], random.sample(B2, 1)[0], random.sample(c_vec[t], 1)[0]) for j in range(m[t])]
        if unique_type == "vertex_distinct":  # Different groups have different C vertices.
            X, Y, Z = random.sample(B1, m[t]), random.sample(B2, m[t]), random.sample(c_vec[t], m[t]) # Sample WITHOUT replacement
            F[t] = [(X[j], Y[j], Z[j]) for j in range(m[t])]
        if unique_type in ["edge", "edge_pairs", "edge-pairs"]:   # Need m edges for (i,j) !! They go together repeatedly with different k values !!
            j_ind = random.sample(range(max(m)), m[t])
            if unique_z:
                Z = random.sample(C, m[t])  # Sample WITHOUT replacement
                F[t] = [(USE_E[j_ind[j]][0], USE_E[j_ind[j]][1], Z[j]) for j in range(m[t])]  # new: sample a subset of m with distinct Z !!!
            else:
                F[t] = [(USE_E[j][0], USE_E[j][1], random.sample(C, 1)[0]) for j in j_ind]  # new: sample a subset of m !!!
        if not unique_type:  # False: non-unique, allow any triplets (i,j,k)
            F[t] = [(random.sample(B1, 1)[0], random.sample(B2, 1)[0], random.sample(C, 1)[0]) for j in range(m[t])]

    # NEW GENERATION SCHEME (FOR CONJECTURE):
    # Generate a set of pairs (x_i, y_i) for pair i  (QU: allow the same vertex to belong to two edges?)
    # For each pair i let C_i \partial D_i \partial {Z} be two sets
    # Connect all x_i < z_j \in D_i   and all y_i < z_j \in C_i
    # For each clause pick all/subset of edges, and for each edge only one/many z_j's?
    return F    # return triplets


# Draw random subsets of edges and try to find a violation of the inequality
# The function finds set collections F_1,..,F_T and G_1,..,G_S of [n]*[n] such that:
# P(intersect_t union_(i,j)<-F_t E_{ij} | intersect_s union_(i,j)<-G_s E_{ij} ) >
# P(intersect_t union_(i,j)<-F_t E_{ij} | intersect_s intersect_(i,j)<-G_s E_{ij} ).
# We assume that [n] = B1 U B2 U B_c and that j <- B_c, and such that i <- B1 for F_t and i <- B2 for G_ss.
# DNF_CNF_order - INTERSECTION of UNIONS (False), as in Theorem (default!!), or UNION of INTERSECTIONS (True)
# unique_type - If "vertex", we must have different vertices. If "edges", we must have triplets with the same (i,j)
# If false, no restriction on the
# Input:
# n - total number of random variables
# B1 - first set (numerator)
# B2 - second set (denominator)
# num_edges - how many pairwise constraint in each intersection/union event set
# num_sets - how many sets of intersection/union
# iters - how many random sets to draw
# full_print - print entire permutations or just their numbers
# triplets - True, do the (i,j,k) in triplets, meaning that for each i in B1 and j in B2 the same k is applied
# unique_type - different types of constraints on the events: LIST HERE ALL POSSIBLE CASES !!
#             (*) False - not constrained. We take any subsets F_1,..,F_T and G_1,..,G_S of [n]*[n]
#             (*) "vertex" - Triplets: For each (i, j, k) the k's for different sets must be different
#             (*) "edge" - Each (i,j) appears only once and in every constraint
#             (*) "edge_B_C_unique" - Each (i,k) or (j,k) appears only once overall
# DNF_CNF_order - False(default): take-INTERSECTION-of-UNIONS, True: take UNION-of-INTERSECTION
def find_random_counter_example_DNF_CNF_inequality(n = 4, B1 = [0, 1], B2 = [],
                                                   num_edges = 3, num_sets = 2, iters = 100,
                                                   full_print=False, triplets=False,
                                                   unique_type=False, DNF_CNF_order=False, randomize=False):
    start_time = time.time()
    #####################################################
    #### Set parameters not fiven/given partially #######
    if not isinstance(num_edges, list):  # all F_t, G_s have the same number of edges
        num_edges_vec_F = [num_edges]*num_sets
    else:  # allow a different number
        num_edges_vec_F = num_edges
    num_edges_vec_G = num_edges_vec_F
    if len(B2) == 0:  # not given
        B2 = B1  # no need for deep copy (we don't alter B)
    #####################################################

    B = list(set(B1).union(B2))
    B_c = [i for i in range(n) if i not in B]  # Complement set for k's. We call it C now
    m = num_edges  # 3  # number of edges in set
    T = num_sets  # 2  # number of sets

    if n > 7:
        perm_iters = 10000
    else:
        perm_iters = -1

    for i in range(iters):
        if randomize:  # here each time take m of each set at random
            num_edges_vec_F = random.choices(range(1, num_edges+1), k=num_sets)  # take uniform sampling
            num_edges_vec_G = random.choices(range(1, num_edges+1), k=num_sets)  # take uniform sampling

        if triplets:  # NEW!!!
            F3 = gen_random_triplets_partial_orders(n, B1, B2, num_edges_vec_F, T, unique_type)  # give vector of m as input !!
            if len(F3) == 0:
                print("No valid triplet, skipping to next one")
                continue
#            print(F3)
            F = [[(x[0], x[2]) for x in y] for y in F3]  # take pairs out of the triplet
            G = [[(x[1], x[2]) for x in y] for y in F3]
        else:
            unique_bool = bool(unique_type)
            if unique_type == "edge_B_C_unique_vertex_in_each_set":
                unique_bool = 2
            F = gen_random_partial_orders(n, B1, num_edges_vec_F, T, B_c, unique_bool)  # change the conditioning and conditioned sets
            G = gen_random_partial_orders(n, B2, num_edges_vec_G, T, B_c, unique_bool)  # any string -> true !!!

#        if DNF_CNF_order:
#            check, num1, num2, denom1, denom2 = check_DNF_CNF_inequality(n, F, G)
#        else:
        check, num1, num2, denom1, denom2 = check_CNF_DNF_inequality(n, F, G, DNF_CNF_order, perm_iters)
        if i%10==0:
            print("\niter:", i, "n=", n, "T=", num_sets, "m=", num_edges_vec_F, " ; ", num_edges_vec_F, "check:", check)
            print("F=", F)
            print("G=", G)

        plus_one_flag = False
        if not check:
            print("Error! Inequality Doesn't Hold! unique_type = ", unique_type)
            print("Error! Inequality Doesn't Hold! n=" + str(n), ", B1 = ", B1, ", B2 = ", B2, " ; B_c=", B_c)
            return output_counterexample(n, F, G, plus_one_flag, full_print, DNF_CNF_order)

    print("Didn't find any counter-example after ", iters, " iters!!!")
    print("Total time:", time.time() - start_time)
    return False


# Print the counter example with many details
def output_counterexample(n, F, G, plus_one_flag=False, full_print=True, DNF_CNF_order=False):
    bool_check, num1, num2, denom1, denom2 = check_CNF_DNF_inequality(n, F, G, DNF_CNF_order)  # Run to get values

    F_print, G_print = copy.deepcopy(F), copy.deepcopy(G)
    if plus_one_flag:
        for i in range(len(F)):
            for j in range(len(F[i])):
                F_print[i][j] = (F_print[i][j][0] + 1, F_print[i][j][1] + 1)
        for i in range(len(G)):
            for j in range(len(G[i])):
                G_print[i][j] = (G_print[i][j][0] + 1, G_print[i][j][1] + 1)

    print("F = ", F_print)
    print("G = ", G_print)

    if full_print and n < 9:  # don't run on large permutations !!
        if DNF_CNF_order:  # unions of intersects, just for printing!!!
            denom1_bad_perms = [p for p in permutations_iter(n) if check_perm_pairs_order_union_of_intersects(p, G)]
            num1_bad_perms = [p for p in denom1_bad_perms if check_perm_pairs_order_union_of_intersects(p, F)]
            denom2_bad_perms = [p for p in
                                permutations_partial_order_iter(n, list(set(sum(G, []))))]  # intersect of intersect
            num2_bad_perms = [p for p in denom2_bad_perms if check_perm_pairs_order_union_of_intersects(p, F)]
        else:  # INTERSECTS of UNIONs, just for printing!!!
            denom1_bad_perms = [p for p in permutations_iter(n) if check_perm_pairs_order_intersect_of_unions(p, G)]
            num1_bad_perms = [p for p in denom1_bad_perms if check_perm_pairs_order_intersect_of_unions(p, F)]
            denom2_bad_perms = [p for p in permutations_partial_order_iter(n, list(
                set(sum(G, []))))]  # ???  intersect of intersect
            num2_bad_perms = [p for p in denom2_bad_perms if check_perm_pairs_order_intersect_of_unions(p, F)]
    else:  # do not compute permutations
        num1_bad_perms = num2_bad_perms = denom1_bad_perms = denom2_bad_perms = []

    if full_print and n < 9 and denom1 < 500 and denom2 < 500:  # Plot for small examples
        print("Bad permutations for denom1:")
        for p in denom1_bad_perms:
            print([x + plus_one_flag for x in p])
        print("Bad permutations for num1:")
        for p in num1_bad_perms:
            print([x + plus_one_flag for x in p])
        print("Bad permutations for denom2:")
        for p in denom2_bad_perms:
            print([x + plus_one_flag for x in p])
        print("Bad permutations for num2:")
        for p in num2_bad_perms:
            print([x + plus_one_flag for x in p])
        print("Num bad perms:", len(num1_bad_perms), len(num2_bad_perms), len(denom1_bad_perms),
              len(denom2_bad_perms), )
    return F, G, denom1_bad_perms, denom2_bad_perms, num1_bad_perms, num2_bad_perms


# Only for triplets,
def find_enumerate_counter_example_DNF_CNF_inequality(n = 4, num_edges = 3, num_sets = 2,
                                                   full_print=False, DNF_CNF_order=False):
    start_time = time.time()
    if n > 7:
        perm_iters = 10000
    else:
        perm_iters = -1
    trip = enumerate_triplets(n, num_edges, num_sets, reduce=True)  # get all triplets
    print("Enumerate", len(trip), " triplets!")
    ctr = 0
    for z in trip:  # Prepare triplets
        T = len(z)
        if ctr % 10 == 0:
            print("\niter:", ctr, "out of ", len(trip), " n=", n, "T=", num_sets, "m=", num_edges)
        F3 = [[0]]*T
        for t in range(T):
            F3[t] = [(j, j+num_edges, z[t][j]+2*num_edges) for j in range(num_edges)]
        F = [[(x[0], x[2]) for x in y] for y in F3]  # take pairs out of the triplet
        G = [[(x[1], x[2]) for x in y] for y in F3]
        check, num1, num2, denom1, denom2 = check_CNF_DNF_inequality(n, F, G, DNF_CNF_order, perm_iters)         # Now check inequality

        if not check:
            print("Error! Inequality Doesn't Hold! n=" + str(n) + " m=" + str(num_edges) + " T=" + str(num_sets))
            return output_counterexample(n, F, G, 1, full_print, DNF_CNF_order)
        ctr += 1

    print("Didn't find any counter-example after full enumeration!")
    print("Total time:", time.time() - start_time)
    return False


# Enumerate all distinct triplets, all possible values of k matching (i,j) pairs
# reduce - eliminate duplicate sets (F_t = F_s) and sort the first set
def enumerate_triplets(n, m, T, reduce=False):
    #    A = list(set(itertools.combinations_with_replacement(range(n-2*m),m)))  # For one subset
    A = list(itertools.product(*[range(n-2*m) for i in range(m)]))  # all (n-2*m)^m combinations
    if reduce:  # sort
        B = [tuple([x]) for x in {tuple(sorted(x)) for x in A}]
    else:
        B = [tuple([x]) for x in copy.deepcopy(A)]

    for t in range(T-1):
        C = [ () for i in range(len(A) * len(B))]  #        C = [None] * len(A) * len(B)
        ctr = 0
        for i in B:
            for j in A:
                C[ctr] = tuple(sorted(i + tuple([j])))
                ctr += 1
        B = list(set(C))  # update B

    if reduce: # remove duplicates
        B = list({tuple(set(x)) for x in B})
    return B


def num_possible_inequalities(k, n, m_F, m_G=[], triplets=False):
    T = len(m_F)
    if triplets:
        return math.comb((n-2*k)**m_F[0]+T-1, T)

    if len(m_G)==0:
        m_G = m_F

    r = 1
    for t in range(len(m_F)):
        r *= math.comb(k, m_F[t]) * math.comb(n-k, m_F[t]) * math.factorial(m_F[t])
    for s in range(len(m_G)):
        r *= math.comb(k, m_G[s]) * math.comb(n-k, m_G[s]) * math.factorial(m_G[s])
    return r / ( math.factorial(len(m_F)) * math.factorial(len(m_G))  )


# New: Check the main inequality of the paper (Theorem 2), while conditioning on x1,2
# New! check the special case with rows and columns (Theorem 2 in the paper), conditioned on x
 # Convention: X_00, ..., X_0(k-1), X_10,..,X_1(k-1),...,x_(n-1)0,...,x_(n-1)(k-1)
def check_conditioned_matrix_CNF_DNF_inequality(n_rows, k_cols, iters = 1000, epsilon = 0.000000001, x_cond = True):
    n = n_rows * k_cols
    P_B23c_and_B13_exact = 0
    ret_flag = True
    if not x_cond:
        P_B13 = 1 / (n_rows-1)**k_cols
        P_B13c = sum([math.comb(n_rows-2, r) * (-1)**r / (r+1)**k_cols for r in range(n_rows-1)])

        P_B23c_and_B13_exact = P_B13 + sum([math.comb(n_rows-2, r) * (-1)**r * ((-1)**(r+1)/ ((r+1)*n_rows))**k_cols for r in range(1, n_rows-1)])  # Wrong prob! (can be negative!

    # Condition all sides on x
    P_B23c_and_B13c_mean, P_B23c_and_B13_mean = 0, 0
    for t in range(iters):  #
        x = np.random.uniform(0, 1, (n_rows,k_cols)) # Randomize array of X
        P_B23c_and_B13c = 1 - np.prod(1-x[0])- np.prod(1-x[1]) + np.prod( 1 - np.maximum(x[0], x[1]))
        P_B23c_and_B13 = np.prod(x[0]) - all(x[0] > x[1]) * np.prod(x[0]-x[1])
        if x_cond:
            P_B13 = np.prod(x[0])
            P_B13c = 1 - np.prod(1-x[0])
        else:
            P_B23c_and_B13c = P_B23c_and_B13c**(n_rows-2)
            P_B23c_and_B13 = P_B23c_and_B13**(n_rows-2)
            P_B23c_and_B13c_mean += P_B23c_and_B13c
            P_B23c_and_B13_mean += P_B23c_and_B13

        if P_B13 * P_B23c_and_B13c > P_B13c * P_B23c_and_B13 + epsilon:  # allow tolerance
            print("Conditioned inequality failes !!!")
            print(x)
            print("P-events:")
            print(P_B13, P_B13c, P_B23c_and_B13c, P_B23c_and_B13)
            ret_flag = False

    P_B23c_and_B13c_mean /= iters
    P_B23c_and_B13_mean /= iters
    print("average prob. inequality: Holds?")
    print(P_B13 * P_B23c_and_B13c_mean < P_B13c * P_B23c_and_B13_mean + epsilon) # Check if inequality holds in aggregate!!!
    print(P_B13, P_B13c, P_B23c_and_B13c_mean, P_B23c_and_B13_mean)
    print("(last one): Exact P_B23c_and_B13_exact: ", P_B23c_and_B13_exact)
    # Another option: condition only two sides

    return ret_flag


def check_matrix_CNF_DNF_inequality_combinatorics(n, k):
    P_B13 = 1 / (n - 1) ** k
    P_B13c = sum([math.comb(n - 2, r) * (-1) ** r / (r + 1) ** k for r in range(n - 1)])

    P_B23c_and_B31 = P_B13 + sum([math.comb(n - 2, r) * (-1) ** r / ((r + 1) * n)**k for r in range(1, n - 1)])  # Wrong prob! (can be negative!
    P_B23c_and_B13c = pareto_P_Bj1c_and_Bj2c_python(k, n)

    if(P_B13 * P_B23c_and_B13c > P_B13c * P_B23c_and_B31):
        print("Error! Violation: Probs:")
    print(P_B13 ,  P_B13c,   P_B23c_and_B13c, P_B23c_and_B31)
    print("Products:")
    print(P_B13 * P_B23c_and_B13c , P_B13c * P_B23c_and_B31)

    return P_B13 * P_B23c_and_B13c <= P_B13c * P_B23c_and_B31

