% simple_diversity_mixin expects the variable "diversity_variables_of_interest"
% to be set as a 1D integer variables array
include "../simple_diversity_mixin.mzn";
% Enumerates values for x from [0, 0] to [2, 2]
array[1..n] of var 0..2: x;
diversity_variables_of_interest = x; % in this model, we care about the values of "x"
% if we cared about other decision variables, too
% we'd still flatten everything out into a single array


%solve maximize diversity_abs;
