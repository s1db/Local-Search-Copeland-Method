 % A student-project matching
include "globals.mzn";
include "../generic_social_welfare.mzn";

int: n; % number of students (which take the role of agents here)
set of int: STUDENT = 1..n;
set of int: AGENTS = STUDENT; % to be consistent with other models

int: p; % number of topics
set of int: PROJECT = 1..p;

int: l; % number of lecturers
set of int: LECTURER = 1..l;
array[PROJECT] of LECTURER: projectToLecturer;

% assign students to topics
array[STUDENT] of var PROJECT: studentToProject;

%-----------------------------------------------------------------------------
% preferences for the students
array[STUDENT, PROJECT] of PROJECT: studentProjectRanks; 

array[STUDENT] of var PROJECT: studentRanksProjectAt;
constraint forall(s in STUDENT) (
   studentRanksProjectAt[s] = studentProjectRanks[s, studentToProject[s]]
);

% convert this to utils -> e.g. rank 1 out of 5 gives Utility 4, rank 5 out of 5 gives Utility 0
array[AGENTS] of var PROJECT: util_per_agent;
constraint forall(s in AGENTS) (
   util_per_agent[s] = p - studentRanksProjectAt[s] 
);


array[STUDENT] of var LECTURER: studentToLecturer;

constraint forall(s in STUDENT) (studentToLecturer[s] = projectToLecturer[studentToProject[s]] );


%-----------------------------------------------------------------------------
% a global cardinality constraint % minimum 1 per project, maximal 3?, same for lecturers

int: minPerLecturer; int: maxPerLecturer;
int: minPerProject; int: maxPerProject;

% some default values for now
minPerLecturer = 0; maxPerLecturer = 4;

% some default values for now
minPerProject = 0; maxPerProject= 3;

constraint global_cardinality_low_up ( studentToLecturer, [l_ | l_ in LECTURER], [minPerLecturer | l_ in LECTURER], [maxPerLecturer | l_ in LECTURER]); 
constraint global_cardinality_low_up ( studentToProject, [p_ | p_ in PROJECT], [minPerProject | p_ in PROJECT], [maxPerProject | p_ in PROJECT]);

ann: random_dec_vars = int_search(studentToProject, input_order, indomain_random, complete);

% solve 
%:: int_search(studentToProject, input_order, indomain_random, complete)
% satisfy;