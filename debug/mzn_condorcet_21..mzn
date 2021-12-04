constraint diversity_variables_of_interest != [11, 19, 15, 21, 0, 3, 8, 13, 0, 3];
constraint diversity_variables_of_interest != [0, 14, 17, 21, 2, 5, 0, 5, 10, 13];
constraint diversity_variables_of_interest != [12, 15, 3, 11, 0, 18, 9, 14, 0, 3];
constraint diversity_variables_of_interest != [0, 5, 17, 21, 2, 13, 0, 5, 10, 13];
constraint diversity_variables_of_interest != [12, 19, 3, 11, 0, 14, 9, 14, 0, 3];
constraint diversity_variables_of_interest != [3, 12, 15, 21, 0, 8, 0, 5, 5, 13];
constraint diversity_variables_of_interest != [0, 3, 13, 21, 2, 18, 8, 13, 0, 5];
constraint diversity_variables_of_interest != [0, 14, 5, 13, 2, 18, 0, 5, 11, 14];
constraint diversity_variables_of_interest != [12, 19, 3, 11, 0, 12, 7, 14, 0, 3];
constraint diversity_variables_of_interest != [0, 5, 13, 21, 2, 18, 0, 5, 10, 13];
constraint diversity_variables_of_interest != [11, 13, 17, 21, 0, 3, 8, 13, 0, 3];
constraint diversity_variables_of_interest != [12, 19, 3, 11, 0, 15, 9, 14, 0, 3];
constraint diversity_variables_of_interest != [11, 19, 15, 21, 0, 3, 7, 13, 0, 3];
constraint diversity_variables_of_interest != [0, 15, 5, 13, 2, 18, 0, 5, 11, 14];
constraint diversity_variables_of_interest != [0, 5, 17, 21, 2, 11, 0, 5, 8, 13];
constraint diversity_variables_of_interest != [0, 5, 14, 21, 2, 18, 0, 5, 10, 13];
constraint diversity_variables_of_interest != [0, 19, 5, 13, 2, 15, 0, 5, 11, 14];
constraint diversity_variables_of_interest != [12, 19, 4, 11, 0, 15, 9, 14, 0, 3];
constraint diversity_variables_of_interest != [12, 14, 3, 11, 0, 18, 9, 14, 0, 3];
constraint diversity_variables_of_interest != [0, 3, 17, 21, 2, 11, 6, 13, 0, 5];
constraint diversity_variables_of_interest != [0, 13, 17, 21, 2, 5, 0, 5, 10, 13];
solve maximize diversity_abs;
old_solutions = [| 3,12,15,21,0,8,0,5,5,13|
12,19,3,11,0,15,9,14,0,3|
0,5,14,21,2,18,0,5,10,13|
12,19,3,11,0,12,7,14,0,3|
0,14,17,21,2,5,0,5,10,13|
0,3,13,21,2,18,8,13,0,5|
12,19,3,11,0,14,9,14,0,3|
0,13,17,21,2,5,0,5,10,13|
12,15,3,11,0,18,9,14,0,3|
0,5,17,21,2,13,0,5,10,13|
11,19,15,21,0,3,8,13,0,3|
0,15,5,13,2,18,0,5,11,14|
12,19,4,11,0,15,9,14,0,3|
0,5,17,21,2,11,0,5,8,13|
11,13,17,21,0,3,8,13,0,3|
0,19,5,13,2,15,0,5,11,14|
12,14,3,11,0,18,9,14,0,3|
0,5,13,21,2,18,0,5,10,13|
11,19,15,21,0,3,7,13,0,3|
0,14,5,13,2,18,0,5,11,14|
0,3,17,21,2,11,6,13,0,5 |];
