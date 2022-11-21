from PartialOrders import *
check_perms_iter = False  # check iterators of permutations
check_union_intersect_count = False  # Check code for counting permutations satisfying unions and intersections of constraints
find_counter = True
check_matrix = True
import matplotlib.pyplot as plt



#######################################################################
if check_perms_iter:  # Example usage: check iterator of permutations
    perms = permutations_iter(4)
    for p in perms:
        print(p)
    perms_0_before_1 = partial_order_extension(4, [(0,1)])
    for q in perms_0_before_1:
        print(q)
#######################################################################

#######################################################################
if check_union_intersect_count:
    F = [[(1,2),(3,4),(1,4),(1,5)], [(2,4),(3,5)], [(2,3),(3,5)], [(4,0),(2,4)]]
    ctr = 0
    for p in any_pair_order_extension(5, F[3]):  # This function seems to work
        ctr += 1
        print(p)
    print(ctr)

    print(count_intersect_posets(6, F, True))  # Intersect of intersect
    print(count_union_of_intersects_posets(6, F, True))  # union of intersects
    print(count_intersect_of_unions_posets(6, F, True))  # INTERSECT of UNIONS THIS Doesn't work ! should be intersect of unions !!
#######################################################################

DNF_CNF_order = False
if DNF_CNF_order:
    find_random_counter_example_DNF_CNF_inequality()  # strong inequality (FALSE!). PROBLEM WITH CODE!!!
    find_random_counter_example_DNF_CNF_inequality(B1 = [0, 1], B2 = [2], iters = 1000, n=5, DNF_CNF_order=True)  # weaker inequality (ALSO FALSE!)
    find_random_counter_example_DNF_CNF_inequality(B1 = [0, 1], B2 = [2], iters = 1000, n=5, DNF_CNF_order=True, triplets=True)  # weaker inequality (ALSO FALSE!)
    find_random_counter_example_DNF_CNF_inequality(B1 = [0, 1], B2 = [2], iters = 1000, n=5, DNF_CNF_order=True, triplets=True, unique_type = "vertex")  # weaker inequality (ALSO FALSE, but couldn't find an example with n=5)
    find_random_counter_example_DNF_CNF_inequality(B1 = [0, 1], B2 = [2], iters = 1000, n=7, DNF_CNF_order=True, triplets=True, unique_type = "vertex")  # weaker inequality (ALSO FALSE!)


# NEW ORDER:
find_random_counter_example_DNF_CNF_inequality(B1 = [0, 1], B2 = [2], iters = 1000, n=5, triplets=True, DNF_CNF_order=False)  # weaker inequality (ALSO FALSE!)
find_random_counter_example_DNF_CNF_inequality(B1 = [0, 1], B2 = [2], iters = 1000, n=5, triplets=True, unique_type = "vertex", DNF_CNF_order=False)  # weaker inequality (ALSO FALSE, but couldn't find an example with n=5)
BAD = find_random_counter_example_DNF_CNF_inequality(B1 = [0, 1], B2 = [2], iters = 1000, n=7, triplets=True, unique_type = "vertex", DNF_CNF_order=False)  # weaker inequality (ALSO FALSE!)
BAD_EDGE = find_random_counter_example_DNF_CNF_inequality(B1 = [0, 1], B2 = [2, 3], num_edges=3, num_sets = 2,
                                                          iters = 1000, n=7,
                                                          triplets=True, unique_type = "edge", DNF_CNF_order=False)  # weaker inequality (ALSO FALSE! CANT FIND COUNTER-EXAMPLE!)

BAD_EDGE = find_random_counter_example_DNF_CNF_inequality(B1 = [0, 1], B2 = [2, 3, 4], num_edges=2, num_sets = 2,
                                                          iters = 1000, n=7,
                                                          triplets=True, unique_type = "edge", DNF_CNF_order=False)  # weaker inequality (ALSO FALSE!)
# 1974 / 2520  =  0.783  <  0.775  =  1302 / 1680
# iter: 2 n= 7 T= 2 m= 2 check: False
# Error! Inequality Doesn't Hold!
# F: [[(0, 6), (1, 6)], [(1, 5), (0, 6)]]
# G: [[(2, 6), (2, 6)], [(2, 5), (2, 6)]]

# Error! Inequality Doesn't Hold! n=7 , B1 =  [0, 1] , B2 =  [2, 3, 4]  ; B_c= [5, 6]



m=3
BAD_EDGE_PAIRS = find_random_counter_example_DNF_CNF_inequality(B1 = range(m), B2 = range(m, 2*m), num_edges=m, num_sets = 2,
                                                          iters = 100, n=7,
                                                          triplets=True, unique_type = "edge_pairs", DNF_CNF_order=False)  # weaker inequality (ALSO FALSE!) CANT FIND COUNTEREXAMPLE

m=3
BAD_EDGE_PAIRS = find_random_counter_example_DNF_CNF_inequality(B1 = range(m), B2 = range(m, 2*m), num_edges=m, num_sets = 3,
                                                                iters = 100, n=8,
                                                                triplets=True, unique_type = "edge_pairs", DNF_CNF_order=False)  # weaker inequality (ALSO FALSE!) CANT FIND!


m = 3  # THIS WORKS ! FOUND A COUNTER EXAMPLE !!!
BAD_EDGE_PAIRS_RAND = find_random_counter_example_DNF_CNF_inequality(
    B1=range(m), B2=range(m,2*m), num_edges=m, num_sets=3,
    iters=300, n=8,
    triplets=True, unique_type="edge_pairs", DNF_CNF_order=False, randomize=True)  # weaker inequality (ALSO FALSE!)
# Error! Inequality Doesn't Hold! unique_type =  edge_pairs
# F =  [[(2, 5)], [(2, 5), (1, 7)], [(1, 6)]]
# G =  [[(4, 5)], [(4, 5), (3, 7)], [(3, 6)]]
# Error! Inequality Doesn't Hold! n=8 , B1 =  range(0, 3) , B2 =  range(3, 6)  ; B_c= [6, 7]
F = [[(1, 4)], [(0, 4)], [(0, 5), (1, 6)]]
G = [[(3, 4)], [(2, 4)], [(2, 5), (3, 6)]]
print(check_CNF_DNF_inequality(7, F, G))

# THIS IS FALSE!!
F2 = [[(0, 4)], [(1, 5)], [(0, 6), (1, 7)]]
G2 = [[(2, 4)], [(3, 5)], [(2, 6), (3, 7)]]
print(check_CNF_DNF_inequality(8, F2, G2))


# THIS IS TRUE!!!
Fm = [[(0, 4), (1, 5)], [(0, 6), (1, 7)]]
Gm = [[(2, 4), (3, 5)], [(2, 6), (3, 7)]]
print(check_CNF_DNF_inequality(8, Fm, Gm))


#################################################################################
#NEW! Keep group sizes the same (no randomize):
m = 3
BAD_EDGE_PAIRS_RAND = find_random_counter_example_DNF_CNF_inequality(
    n=8, B1=range(m), B2=range(m,2*m),
    num_edges=m, num_sets=3, iters=100,
    triplets=True, unique_type="edge_pairs")  # weaker inequality (ALSO FALSE!)
#################################################################################


# New: bad vertex
m = 3
start_time = time.time()
BAD_VERTEX_UNIQUE_RAND = find_random_counter_example_DNF_CNF_inequality(B1 = range(m), B2 = range(m,2*m), num_edges=m, num_sets = 2,
                                                                     iters = 100, n=12,
                                                                     triplets=True, unique_type = "vertex_distinct", DNF_CNF_order=False, randomize=True)  # weaker inequality (ALSO FALSE!)
print("Total time:", time.time()-start_time)


m, T = 2, 3
# We need B1*C >= m*T
BAD_UNIQUE_EDGE = find_random_counter_example_DNF_CNF_inequality(B1 = range(m+1), B2 = range(m+1,2*m+2), num_edges=m, num_sets = T,
                                                                     iters = 100, n=2*m + T+4,
                                                                     triplets=False, unique_type = "edge_B_C_unique", DNF_CNF_order=False, randomize=True)  # weaker inequality (ALSO FALSE!)


















F3 = gen_random_triplets_partial_orders(n=8, B1 = range(m), B2 = range(m,2*m),  m=2, T=2, unique_type="vertex_distinct")  # must have n >= 2*T*m

# Try a particular bad example
# F_bad = [[(0, 3), (0, 2), (1, 3)], [(1, 3), (0, 3)]]
F_bad = [[(0, 3),  (1, 3)]]
G_bad = [[(1, 3)], [(1, 3), (1, 2)]]
print(check_DNF_CNF_inequality(4, F_bad, G_bad))


F_b2 = [[(0, 4), (1, 4)], [(1, 4), (1, 3), (0, 3)]]
G_b2 = [[(2, 3), (2, 4)], [(2, 4)]]
print(check_CNF_DNF_inequality(5, F_b2, G_b2))


F_b3 =  [[(1, 2), (0, 4)], [(1, 2), (0, 2)]]
G_b3 =  [[(3, 2), (3, 4)], [(3, 2)]]
print(check_CNF_DNF_inequality(7, F_b3, G_b3))

F_bad_diff = [[(0, 3)], [(0, 6)]]
#G_bad_diff = [[(1, 3)], [(1, 4), (2, 5)] ]
G_bad_diff = [[(1, 3)], [(1, 4), (1, 5)] ]
print(check_CNF_DNF_inequality(7, F_bad_diff, G_bad_diff))

m, T = 2, 2
# We need B1*C >= m*T
BAD_UNIQUE_EDGE_DIFF = find_random_counter_example_DNF_CNF_inequality(B1 = [0], B2 = [1], num_edges=m, num_sets = T,
                                                                     iters = 100, n=6,
                                                                     triplets=False, unique_type = "edge_B_C_unique", DNF_CNF_order=False, randomize=True)  # weaker inequality (ALSO FALSE!)
n=6
F =  [[(0, 2), (0, 3)], [(0, 4), (0, 5)]]
G =  [[(1, 2), (1, 3)], [(1, 4), (1, 5)]]
print(check_CNF_DNF_inequality(n, F, G))
output_counterexample(n, F, G)
print(check_CNF_DNF_inequality(n, [F[1]], [G[1]]))

# Try to figure out how the hell is the above inequality true
denom1_bad_perms = [p for p in permutations_iter(n) if check_perm_pairs_order_intersect_of_unions(p, G)]
num1_bad_perms = [p for p in denom1_bad_perms if check_perm_pairs_order_intersect_of_unions(p, F)]
denom2_bad_perms = [p for p in permutations_partial_order_iter(n, list(
    set(sum(G, []))))]  # ???  intersect of intersect
num2_bad_perms = [p for p in denom2_bad_perms if check_perm_pairs_order_intersect_of_unions(p, F)]

d1 = {tuple(x) for x in denom1_bad_perms}
d2 = {tuple(x) for x in denom2_bad_perms}
d1_and_not2 = d1 - d2
n1 = {tuple(x) for x in num1_bad_perms}
n2 = {tuple(x) for x in num2_bad_perms}
n1_and_not2 = n1 - n2


d1_and_not2_no_sym = {x for x in d1_and_not2 if x.index(2)<x.index(3) and x.index(4)<x.index(5) and min(x.index(2),x.index(3))<min(x.index(4),x.index(5))}
n1_and_not2_no_sym = n1_and_not2.intersection(d1_and_not2_no_sym)

d1_and_not2_no_sym_notnum = d1_and_not2_no_sym - n1_and_not2_no_sym

# Numerator
for p in n1_and_not2_no_sym:
    print([x for x in p])
# Complement
for p in d1_and_not2_no_sym_notnum:
    print([x for x in p])

# NEW: Check matrix inequality:
if check_matrix:
    print(check_matrix_CNF_DNF_inequality(3, 3))



m, T = 2, 3
# We need B1*C >= m*T
BAD_UNIQUE_EDGE_DIFF = find_random_counter_example_DNF_CNF_inequality(B1 = [0,1], B2 = [2,3], num_edges=m, num_sets = T,
                                                                     iters = 100, n=8,
                                                                     triplets=False, unique_type = "edge_B_C_unique_vertex_in_each_set", DNF_CNF_order=False, randomize=True)  # weaker inequality (ALSO FALSE!)



m, T = 3, 3
# We need B1*C >= m*T
BAD_UNIQUE_EDGE_DIFF_ONE_B = find_random_counter_example_DNF_CNF_inequality(B1 = [0,1,2], B2 = [0,1,2], num_edges=3, num_sets = 3,
                                                                     iters = 100, n=8,
                                                                     triplets=False, unique_type = "edge_B_C_unique_vertex_in_each_set", DNF_CNF_order=False, randomize=True)  # weaker inequality (ALSO FALSE!)


F = [[(0, 3)]]
G = [[ (1, 3), (2, 3) ], [(1, 4), (2,3)]]

F = [[(0, 2)]]
G = [[ (0, 3), (1, 3) ]]

F =  [[(0, 3), (1, 3)], [(0, 4), (1, 6)], [(1, 3)]]
G =  [[(0, 5), (2, 7)], [(2, 3)], [(0, 3)]]
print(check_CNF_DNF_inequality(8, F, G))

F =  [ [(0, 4), (1, 6)], [(1, 3)]]
G =  [[(0, 5), (2, 7)], [(2, 3)], [(0, 3)]]
print(check_CNF_DNF_inequality(8, F, G))


# Simpler version:
F = [[(0, 5)], [(0, 3)]]
G = [[(1, 4), (2, 6)], [(1, 3)]]
print(check_CNF_DNF_inequality(7, F, G))
(False, 861, 336, 2100, 840)

F = [[(1, 5)], [(1, 6)] ]
G = [[(3, 5)], [(3, 7), (4, 8)]]
print(check_CNF_DNF_inequality(9, F, G))



F =  [ [(1, 4)], [(1, 4), (2, 4)] ]
G =  [ [(1, 3)], [(1, 3), (2, 3)] ]
print(check_CNF_DNF_inequality(8, F, G))


# Try big examples
n = 10
FF = [[(0, 4), (1, 4), (4,5)], [(1, 4), (1, 3), (0, 3)], [(4,5), (5,7), (7,8)], [(5,9), (3,9)]]
start_time = time.time()
cc = count_intersect_of_unions_posets(n, FF)
print("Count: ", cc, " fraction:", cc / math.factorial(n))
print("Time took:", time.time()-start_time)



# NEW! Enumerate all possibilities with fixed size !!!
find_enumerate_counter_example_DNF_CNF_inequality(n = 9, num_edges = 4, num_sets = 3)




# New: use combinatoircs to verify matrix inequality:
max_n = 100
max_k = 100
P_left = np.zeros( (max_n, max_k))
P_right = np.zeros( (max_n, max_k))
viol = False
for n in range(2, max_n):
    print("Try n = ", n)
    for k in range(2, max_k):
        check_vec = check_matrix_CNF_DNF_inequality_combinatorics(n, k)
        P_left[n,k], P_right[n,k] = check_vec[1],  check_vec[2]
        if not check_vec[0]:
            print("Violation of inequality!!!!, n=", n, " k=", k)
            viol = True
            break
print("Violated?", viol)
P_diff = P_right - P_left
P_ratio = P_right / P_left - 1


# Weird observation: E_Z1Z2_n_k_plus_1 = P_Bj1c_Bj2c_n_k. Why? ONLY FOR n=3 !!! not universal!
for n in range(2, 10):
    for k in range(2, 10):
        print(pareto_P_Bj1c_and_Bj2c_python(k,n) - pareto_E_Z1Z2_python(k+1,n))


# Check manually
n=4
k=3
c = 0
P_B13c = sum([math.comb(n - 2, r) * (-1) ** r / (r + 1) ** k for r in range(n - 1)])
P_B31 = 1/ ((n-1)**k)

n,k=3,3
ctr = 0
P_B13c_B23c = 0
P_B31_B23c = 0
for p in permutations_iter(n):
    print("Run p:", ctr)
    ctr += 1
    for q in permutations_iter(n):
        for s in permutations_iter(n):
            B13c, B23c, B31 = True, True, True # This is intersection over all j's
            for j in range(2,n):
                if p[j] > p[0] and q[j] > q[0] and s[j] > s[0]:
                    B13c = False
                if p[j] > p[1] and q[j] > q[1] and s[j] > s[1]:
                    B23c = False
                if p[j] > p[0] or q[j] > q[0] or s[j] > s[0]:
                    B31 = False
            P_B13c_B23c += (B13c) * (B23c)
            P_B31_B23c += B31 * (B23c)
print(P_B13c_B23c, P_B31_B23c)

print(P_B31, P_B13c)

#            print(p, q, s)
print(c)

print(P_B13c * P_B31_B23c)
print(P_B31 * P_B13c_B23c)

plt.plot(P_ratio[3])

n, k = 3, 2
check_conditioned_matrix_CNF_DNF_inequality(n, k, x_cond = True, y_cond = False)


