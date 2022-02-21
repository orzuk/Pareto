from PartialOrders import *

check_perms_iter = False


if check_perms_iter: # Example usage:
    perms = permutations_iter(4)
    for p in perms:
        print(p)

    perms_0_before_1 = partial_order_extension(4, [(0,1)])
    for q in perms_0_before_1:
        print(q)


F = [[(1,2),(3,4), (1,4), (1,5)], [(2,4),(3,5)], [(2,3),(3,5)], [(4,0),(2,4)]]

print(count_intersect_posets(6, F))
print(count_union_posets(6, F))


find_random_counter_example_DNF_CNF_inequality()  # strong inequality (FALSE!)


find_random_counter_example_DNF_CNF_inequality(B_1 = [0, 1], B_2 = [2], iters = 1000)  # strong inequality (FALSE!)


# F_bad = [[(0, 3), (0, 2), (1, 3)], [(1, 3), (0, 3)]]
F_bad = [[(0, 3),  (1, 3)]]
G_bad = [[(1, 3)], [(1, 3), (1, 2)]]
print(check_DNF_CNF_inequality(4, F_bad, G_bad))




# Another example: add structure :

