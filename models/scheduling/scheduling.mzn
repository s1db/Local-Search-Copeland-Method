include "scheduling_core.mzn";
include "../simple_diversity_mixin.mzn";

diversity_variables_of_interest = [start[j,t] | j in JOBS, t in TASKS] ;
%diversity_variables_of_interest = util_per_agent;

%solve
%:: worst_case
%satisfy;
