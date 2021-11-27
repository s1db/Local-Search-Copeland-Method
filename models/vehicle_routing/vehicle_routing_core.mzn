% Use this editor as a MiniZinc scratch book
include "../generic_social_welfare.mzn";
set of int: Site;
Site: hotel;
array[Site,Site] of int: distance;

array[Site] of int: durationOfVisit;
array[Site] of int: siteOpens;
array[Site] of int: siteCloses;

array[Site] of var Site: next;

include "subcircuit.mzn";
constraint subcircuit(next);

array[Site] of var 0..infinity: arrivalTime;
array[Site] of var 0..infinity: startOfVisit;

constraint arrivalTime[next[hotel]] = distance[hotel,next[hotel]];
constraint forall (i in Site where i != hotel /\ i != next[i]) (
  arrivalTime[next[i]] = startOfVisit[i] + durationOfVisit[i] + distance[i,next[i]]
);

constraint forall (i in Site where i != next[i]) (
   startOfVisit[i] = max(arrivalTime[i], siteOpens[i])
/\ startOfVisit[i]+durationOfVisit[i] <= siteCloses[i]
);

constraint forall (i in Site where i = next[i]) (
  startOfVisit[i]=0 /\ arrivalTime[i]=0
);

var int: travel_time ;
constraint travel_time = sum (i in Site where i != next[i]) (distance[i,next[i]]);

set of int: Passenger;
set of int: AGENTS = Passenger; 

array[Passenger,Site] of int: site_score;
array[Passenger] of int: preferred_n_sites;
array[Passenger] of int: preferred_travel_time;
array[Passenger,1..3] of int: preference_weights;

array[Passenger,1..3] of var int: passenger_scores 
= 
array2d(Passenger,1..3,
[ if i=1 then sum (s in Site where s != hotel /\ s != next[s]) (site_score[p,s])
  elseif i=2 then (preferred_n_sites[p]-abs(preferred_n_sites[p] - sum (s in Site where s != hotel) (s != next[s])))
  else min(preferred_travel_time[p] - travel_time,0)
  endif
| p in Passenger, i in 1..3]);

array[Passenger] of var int: score =
[   preference_weights[p,1] * passenger_scores[p,1]
  + preference_weights[p,2] * passenger_scores[p,2]
  + preference_weights[p,3] * passenger_scores[p,3]
  | p in Passenger ];

array[AGENTS] of var int: util_per_agent ;
constraint forall(s in AGENTS) (
   util_per_agent[s] = score[s]
);

ann: random_dec_vars = int_search(startOfVisit, input_order, indomain_random, complete);

%include "../diversity_mixin.mzn";
%array[int] of var Site: diversity_x ::add_to_output = next;
