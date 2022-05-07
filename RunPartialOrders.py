from PartialOrders import *
check_perms_iter = False  # check iterators of permutations
check_union_intersect_count = False  # Check code for counting permutations satisfying unions and intersections of constraints
find_counter = True
check_matrix = True

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


find_random_counter_example_DNF_CNF_inequality()  # strong inequality (FALSE!). PROBLEM WITH CODE!!!
find_random_counter_example_DNF_CNF_inequality(B_1 = [0, 1], B_2 = [2], iters = 1000, n=5)  # weaker inequality (ALSO FALSE!)
find_random_counter_example_DNF_CNF_inequality(B_1 = [0, 1], B_2 = [2], iters = 1000, n=5, triplets=True)  # weaker inequality (ALSO FALSE!)
find_random_counter_example_DNF_CNF_inequality(B_1 = [0, 1], B_2 = [2], iters = 1000, n=5, triplets=True, unique_type = "vertex")  # weaker inequality (ALSO FALSE, but couldn't find an example with n=5)
find_random_counter_example_DNF_CNF_inequality(B_1 = [0, 1], B_2 = [2], iters = 1000, n=7, triplets=True, unique_type = "vertex")  # weaker inequality (ALSO FALSE!)


# NEW ORDER:
find_random_counter_example_DNF_CNF_inequality(B_1 = [0, 1], B_2 = [2], iters = 1000, n=5, triplets=True, DNF_CNF_order=False)  # weaker inequality (ALSO FALSE!)
find_random_counter_example_DNF_CNF_inequality(B_1 = [0, 1], B_2 = [2], iters = 1000, n=5, triplets=True, unique_type = "vertex", DNF_CNF_order=False)  # weaker inequality (ALSO FALSE, but couldn't find an example with n=5)
BAD = find_random_counter_example_DNF_CNF_inequality(B_1 = [0, 1], B_2 = [2], iters = 1000, n=7, triplets=True, unique_type = "vertex", DNF_CNF_order=False)  # weaker inequality (ALSO FALSE!)
BAD_EDGE = find_random_counter_example_DNF_CNF_inequality(B_1 = [0, 1], B_2 = [2, 3], num_edges=3, num_sets = 2,
                                                          iters = 1000, n=7,
                                                          triplets=True, unique_type = "edge", DNF_CNF_order=False)  # weaker inequality (ALSO FALSE!)

BAD_EDGE = find_random_counter_example_DNF_CNF_inequality(B_1 = [0, 1], B_2 = [2, 3, 4], num_edges=2, num_sets = 2,
                                                          iters = 1000, n=7,
                                                          triplets=True, unique_type = "edge", DNF_CNF_order=False)  # weaker inequality (ALSO FALSE!)


m=3
BAD_EDGE_PAIRS = find_random_counter_example_DNF_CNF_inequality(B_1 = range(m), B_2 = range(m,2*m), num_edges=m, num_sets = 2,
                                                          iters = 1000, n=8,
                                                          triplets=True, unique_type = "edge_pairs", DNF_CNF_order=False)  # weaker inequality (ALSO FALSE!)


# Try a particular bad example
# F_bad = [[(0, 3), (0, 2), (1, 3)], [(1, 3), (0, 3)]]
F_bad = [[(0, 3),  (1, 3)]]
G_bad = [[(1, 3)], [(1, 3), (1, 2)]]
print(check_DNF_CNF_inequality(4, F_bad, G_bad))



# NEW: Check matrix ineqaulity:
if check_matrix:
    print(check_matrix_CNF_DNF_inequality(3, 3))
