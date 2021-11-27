 % A student-project matching
include "../simple_diversity_mixin.mzn";
include "project_assignment_core.mzn";

%----------------------------------------------------------------------------
% connecting the model hook from the diversity mixin to this model
diversity_variables_of_interest = studentToProject; % all other variables depend on this assignment   