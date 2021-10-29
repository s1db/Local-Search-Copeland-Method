% ---------------------------------------------------------------------------------------
% simple_diversity_mixin expects the variable "diversity_variables_of_interest"
% to be set as a 1D integer variables array

% Inputs:
%    - diversity_variables_of_interest ... has to be tied to meaningful variables in a
%                                          model
%    - old_solutions                   ... data input of n copies of the values 
%                                          of the variables of interest from previous 
%                                          solutions
% Outputs
%    - div_count_changed               ... the number of changed values over all past solutions
%    - div_abs_changed                 ... the sum of absolute values that changed over all past solutions
%
% Can be used for diversity optimization as follows:
%    - solve maximize div_count_changed
%    - solve maximize div_abs_changed;
% ---------------------------------------------------------------------------------------
array[int] of var int: diversity_variables_of_interest; 
array[int, index_set(diversity_variables_of_interest)] of int: old_solutions;
array[index_set_1of2(old_solutions), index_set(diversity_variables_of_interest)] of var bool: div_changed; 
array[index_set_1of2(old_solutions), index_set(diversity_variables_of_interest)] of var int: div_abs_dists; 

constraint forall (i in index_set_1of2(old_solutions), j in index_set_2of2(old_solutions)) (
  div_changed[i, j] = (old_solutions[i, j] != diversity_variables_of_interest[j])  /\
  div_abs_dists[i, j] = abs(old_solutions[i, j] - diversity_variables_of_interest[j])
);

% ---------------------------------------------------------------------------------------
% Offered objective functions as decision variables
var int: diversity_abs = sum(div_abs_dists);

int: div_max_changes = if index_set_1of2(old_solutions) == {} then 0 else max(index_set_1of2(old_solutions)) * max(index_set_2of2(old_solutions)) endif;
var 0..div_max_changes: div_count_changed =  sum(div_changed);

% ---------------------------------------------------------------------------------------
% Hard constraints to enforce new solutions (in order for an outer python search to terminate

constraint forall(i in index_set_1of2(old_solutions)) (
   diversity_variables_of_interest != [ old_solutions[i,j] | j in index_set_2of2(old_solutions)]
);