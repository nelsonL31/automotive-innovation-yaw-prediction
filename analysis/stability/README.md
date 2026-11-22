# Straight-running stability at fixed axle stiffness

`fixed_stiffness_jacobians.csv` records the 48 continuous-time linearisations in Section 4.2, covering eight front/rear axle stiffness settings at six forward speeds (2, 5, 10, 15, 20 and 25 m/s). Each row uses zero steering, zero lateral velocity and zero yaw rate. State order is lateral velocity (m/s), then yaw rate (rad/s). The four matrix entries follow this order. A negative largest real eigenvalue indicates local asymptotic stability of this fixed-parameter equilibrium.

The values were extracted from the frozen analysis output `kit_stability.json` (SHA-256 `d5c44b5579f80dc2e40d9539d08bc7ff3173dd9ae4e81b046645fba48965e781`). Its recorded agreement flag for the analytical and automatic-differentiation Jacobians was true. The speed, stiffnesses, matrix entries, largest real eigenvalue and stability flag are archived values. The determinant column was calculated from the archived matrix.

For a speed v, the smoothing factor is h = v² / [(v² + 2²) sqrt(v² + 0.5²)]. Define k = lr Cr − lf Cf, with Cf and Cr the positive front and rear axle stiffnesses. The four Jacobian entries are a11 = −(Cf + Cr) h / m, a12 = k h / m − v, a21 = k h / Iz, and a22 = −(lf² Cf + lr² Cr) h / Iz. The mass m, yaw inertia Iz and axle distances lf and lr come from `models/kit_contract.json`.

Independent recomputation matches the archived matrix entries to a maximum absolute difference of 5.68e-14. The corresponding closed-form determinant, Cf Cr (lf + lr)² h² / (m Iz) + k v h / Iz, matches the matrix determinants within 3.64e-12 s⁻². For Cf = 468016 N/rad and Cr = 23400.8 N/rad, solving for zero determinant by bisection between 5 and 10 m/s gives 7.65034 m/s. This speed applies to the stated axle stiffness pair. Section 4.3 separately analyses disturbances propagated through a learned stiffness estimate.

CSV SHA-256: `e972c6809dd3354caaa2364d3a073d436435103b8cae9477d731e45c7f747d72`.
