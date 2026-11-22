# Scale-racing closed-loop outcomes

`scale_racing_closed_loop_by_seed.csv` gives the twelve offline simulation attempts reported in Section 5.3 and Table 4. Each attempt used a final 1,000-epoch predictor trained on DYN-PP-ETHZ, the same model predictive controller and one of three matched seeds (11, 22, 33).

A completed lap passed the ordered progress and finish-gate checks. The recorded lap time is the finish sample time in seconds. An empty time denotes an incomplete attempt. For each attempt, the elapsed run contributes its actual executed step count. On a stopped attempt, boundary and timing counts cover only those steps. Boundary excursions count episodes, while outside samples count individual samples. A completed lap can include excursions. The timing column gives the archived deadline-miss count for the 20 ms control period. Incomplete attempts retain the recorded stopping cause.

The data were extracted from the four-method simulation evaluation archive `evaluation_closed/results.json` (SHA-256 `da5afeb20a3c66ca516005cc54cea8c84257bd27aba37eea2421ff7d7e03645f`). The extraction used `arm`, `seed`, `completed`, `lap_time_s`, `executed_steps`, `deadline_misses`, `track_violation_episodes`, `outside_samples`, the stopping status and the per-model checkpoint digest. In this archive, `DDM`, `ode`, `pde` and `joint` identify the reproduced PGNN, ODE-PINN, PDE-PINN and joint ODE–PDE PINN. The simulation log names `Maximum_Iterations_Exceeded` and `Ambiguous/discontinuous centerline projection` map to the two stopping categories in the CSV.

The twelve rows include six complete laps and six interrupted attempts. The reproduced PGNN completed two laps, ODE-PINN three, PDE-PINN one and the joint model zero. The three ODE-PINN attempts recorded 0, 1 and 2 boundary excursions and 269/285, 279/295 and 277/293 steps over the deadline, in seed order.

CSV SHA-256: `90aba7dbba3e9cc431af3b0c15f7b82d2526a9c70d363fba697c5727875dd1fb`.
