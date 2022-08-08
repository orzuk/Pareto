# Pareto
Computing statistics for Pareto-optimal vectors


To check whether certain correlation inequalities hold, use the function 'find_enumerate_counter_example_DNF_CNF_inequality'. 
For example: <br>
\>\>\> find_enumerate_counter_example_DNF_CNF_inequality(n = 8, num_edges = 3, num_sets = 3)

To find counter-examples to certain inequalities randomly, use the function 'find_random_counter_example_DNF_CNF_inequality'. 
For example: <br>
\>\>\> find_random_counter_example_DNF_CNF_inequality(B1 = [0,1,2], B2 = [0,1,2], num_edges=3, num_sets = 3, iters = 100, n=8,
                                  triplets=False, unique_type = "edge_B_C_unique_vertex_in_each_set", DNF_CNF_order=False, randomize=True) 
