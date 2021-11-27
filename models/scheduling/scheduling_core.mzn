%-----------------------------------------------------------------------------
% Job-shop scheduling where jobs belong to agents
%
% Alexander Schiendorfer
% 2021-11-27
%
%-----------------------------------------------------------------------------
include "globals.mzn";

include "../generic_social_welfare.mzn";

%-----------------------------------------------------------------------------
% Specification

int: n_agents;
set of int: AGENTS = 1..n_agents;

% Use this editor as a MiniZinc scratch book
int: lastT = sum(j in JOBS, t in TASKS)(dur[j,t]);

set of int: TIME = 0..lastT;

int: n_jobs;
set of int: JOBS = 1..n_jobs;
int: n_tasks;

set of int: TASKS = 1..n_tasks;

array[JOBS,TASKS] of int: dur;

% actual decision variables
array[JOBS,TASKS] of var TIME: start;
array[JOBS,TASKS] of var TIME: end;

constraint forall(j in JOBS, t in TASKS)(
  end[j,t] = start[j,t] + dur[j,t]
);

% machines 
set of int: MACHINES = 1..3;
array[JOBS,TASKS] of MACHINES: taskToMach;

constraint 
  forall(j1, j2 in JOBS, t1, t2 in TASKS where 
                  j1 != j2
               /\ taskToMach[j1, t1] = taskToMach[j2, t2]) (
                  end[j1, t1] <= start[j2, t2] \/
                  end[j2, t2] <= start[j1, t1]
                 ); 

% task ownership
array[JOBS] of AGENTS: owner;

% precedences 
constraint forall(j in JOBS, t in 1..max(TASKS)-1) (
  end[j,t] <= start[j,t+1]
);

% objective 
array[AGENTS] of var TIME: util_per_agent;

var TIME: makespan; 
constraint makespan = max(j in JOBS, t in TASKS)(end[j,t]);
TIME: max_makespan;
constraint makespan <= max_makespan;

constraint forall(a in AGENTS) (util_per_agent[a] = lastT - max(j in JOBS, t in TASKS where owner[j] = a)(end[j,t]));

ann: random_dec_vars = int_search(start, input_order, indomain_random, complete);

%solve minimize makespan;