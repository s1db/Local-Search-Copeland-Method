constraint diversity_variables_of_interest != [12, 19, 3, 11, 0, 15, 9, 14, 0, 3];
constraint diversity_variables_of_interest != [3, 12, 15, 21, 0, 8, 0, 5, 5, 13];
solve maximize diversity_abs;
old_solutions = [| 3,12,15,21,0,8,0,5,5,13|
12,19,3,11,0,15,9,14,0,3 |];
