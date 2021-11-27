
% predefined search annotations to try out 
ann: worst_case = int_search([social_welfare], input_order, indomain_min, complete);
ann: best_case = int_search([social_welfare], input_order, indomain_max, complete);
ann: random_welfares = int_search([social_welfare], input_order, indomain_random, complete);

% social welfare as a proxy for Copeland scores to try out 

var int: social_welfare;
constraint social_welfare = sum(util_per_agent);
