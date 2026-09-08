# Revision solver verification: predeclared protocol v1

Protocol fixed before the first numerical execution on 2026-09-08. This file
describes new verification cases, not measured results. Actual results and the
SHA-256 of this protocol are written by `scripts/verify_revision_solver.py`.

## Purpose and implementation inspected

The existing production `simulate_cure_2d` time loop is used without patches,
monkeypatches, replacement operators, or test-only integration routines. Its
temperature update is backward Euler, with cell-centred two-point finite-volume
diffusion, harmonic interface conductance, and half-cell conduction plus Robin
convection resistances. Therefore first-order temporal convergence is expected
for the heat equation. Second-order spatial convergence is expected for the
uniform rectangular grids used below. The cure update is explicit RK4 inside
the production Picard coupling loop. The synthetic temperature-independent cure
case isolates the RK4 order and the discrete reaction-heat balance.

Every exact evaluator below uses continuous equations and elementary functions.
It does not import production flux assembly, residuals, linear systems, or the
production cure-rate evaluator. Grid and kinetic dataclasses are inputs to the
production calculation, not reference solvers. The cases are original synthetic
verification definitions and use no upstream ResFNO data or checkpoints.

## Common constants and error definitions

Domain: x in [0, 0.2] m; z in [0, 0.1] m. Density 1000 kg/m3 and specific heat
1000 J/(kg K), hence volumetric heat capacity C=1e6 J/(m3 K). Ambient and baseline
temperature T0=300 K; perturbation amplitude A=10 K; kx=2 W/(m K).

Report volume-weighted absolute RMS L2 temperature error in K, Linf error in K,
and relative L2 with denominator `T_exact - T0`, not the large absolute-Kelvin
baseline. For the cure case the exact mean reaction-heating component is also
part of this perturbation denominator. Errors are reported at each saved
noninitial time; convergence acceptance is at the final time. Observed order is
`log(error_coarse/error_fine)/log(2)`, for each adjacent refinement pair.

## Case N: nonconstant x-z insulated transient eigenmode

Set kz=0.5 W/(m K), all four Robin coefficients to zero, composite mask false,
and reaction heat source zero. Define

`T*(x,z,t) = 300 + 10 cos(pi*x/0.2) cos(pi*z/0.1) exp(-lambda*t)`

with `lambda = [2*(pi/0.2)^2 + 0.5*(pi/0.1)^2]/1e6`.
Both spatial derivatives are nonzero in the interior. The derivative normal to
every boundary is zero. This is a solution of the continuous homogeneous
anisotropic heat equation. Initialize the solver with its cell-centre values.

Spatial study: grids (nx,nz)=(12,8),(24,16),(48,32), saved times 0,50,100 s,
maximum step 0.02 s. On the finest grid repeat with 0.01 s to assess remaining
time error. Predeclared gates: adjacent L2 spatial orders in [1.8,2.2]; final
finest relative L2 <2e-4 and Linf <0.002 K; the difference caused by halving the
time step must be <10% of the finest-grid error relative to the exact solution.

Temporal study: fixed (96,64) grid, steps 20,10,5 s, saved times 0,100 s. (Using
only endpoints prevents output intervals from silently changing a 20 s step.)
Repeat the finest time step on (192,128) to bound the spatial contribution.
Predeclared gates: adjacent L2 temporal orders in [0.85,1.15]; final finest
relative L2 <4e-4; change in final scalar L2 error on spatial refinement <10% of
the coarse-grid finest-time-step error. This explicitly checks the spatial
floor; the continuous PDE solution remains the reference at all resolutions.

## Cases R and I: steady x-z manufactured fields, four Robin boundaries

Let `f(x)=1+Bx*x*(Lx-x)`, Bx=50 1/m2. Introduce resistance coordinate
`u(z)=integral_0^z 1/kz(s) ds`, U=u(Lz), and
`g(z)=1+Bz*u(z)*(U-u(z))`, where `Bz=2/U^2`.
Then `T*(x,z)=300+10*f(x)*g(z)` is continuous and nonconstant in both axes.
The analytical heat flux `-kz*dT*/dz=-10*f*Bz*(U-2u)` is continuous even
when kz jumps. Define the time-independent volumetric source from the
continuous PDE:

`q*(x,z) = 20*[kx*Bx*g(z) + (Bz/kz(z))*f(x)]` W/m3.

All four boundaries use the same 300 K ambient. The Robin coefficients are
`hx=kx*Bx*Lx` on left/right and `hz=Bz*U` on lower/upper boundaries. These
follow directly from `-k*dT*/dn = h*(T*-300)`. No discrete residual generates
the source. The full production loop starts from T* and relaxes to the discrete
steady state with the prescribed q*.

Case R: kz=0.5 everywhere, no composite cells. Case I: kz=1 below z=0.05 m and
kz=0.25 above it; the interface aligns with a cell face on every grid. The upper
layer is marked composite solely to exercise production interface diagnostics;
initial alpha=1 and reaction heat coefficient=0 keep this an inert conduction
case. The interface has nonzero heat flux and unequal one-sided T gradients.

Spatial studies use (12,8),(24,16),(48,32), saved times 0,100000,200000 s,
maximum step 1000 s. Predeclared gates for each case: adjacent final L2 orders
in [1.8,2.2], finest relative L2 <0.002, finest Linf <0.05 K; maximum field
change from the penultimate to final time <1e-7 K to exclude a significant
remaining transient. This is steady code verification, not transient Robin
or transient interface accuracy evidence.

## Case C: exact special-case reaction plus true x-z heat diffusion

Use the Case N grid and boundaries, mark every cell composite, and set reaction
heat coefficient S=2e7 J/m3 per unit alpha. In the existing CureKinetics input
set A=0.02 /s, delta_E=0, M=0, N=1, C=0, denominator_offset=1. This gives
`dalpha/dt=gamma*(1-alpha)`, gamma=0.01 /s, independent of temperature.
With uniform initial alpha0=0.1:

`alpha*(t)=1-0.9*exp(-0.01*t)`;

`T*(x,z,t)=T_N*(x,z,t)+(S/C)*(alpha*(t)-0.1)`.

Use (12,8), saved times 0,100 s, steps 20,10,5 s. Predeclared gates: adjacent
alpha Linf convergence orders in [3.8,4.4], finest alpha Linf <1e-6, finest
temperature Linf <0.03 K, and the maximum error in the exact mean reaction-heat
identity `mean(T)-300=(S/C)*(mean(alpha)-0.1)` <1e-9 K. This exercises the
production cure update, source increment, and Picard loop. It does not verify
temperature-dependent AS4/8552 kinetics or validate real material parameters.

## Shared acceptance, execution, and outputs

All runs must finish with finite values and converged production coupling.
Source/reference fields must vary by >0.01 K along each spatial axis. Preserve
each case's full numerical/exact arrays, metadata, diagnostics, errors, and
failure exception if any. Do not silently drop a failed case or overwrite an
existing result directory. A changed protocol requires a new version and a new
output directory with the earlier outcomes retained.

Run with a pinned environment from the repository root:

```text
python scripts/verify_revision_solver.py --output-dir outputs/revision_solver_verification/RUN_ID --environment-label ENVIRONMENT_DESCRIPTION
python -m pytest -q tests/scientific/test_revision_solver_verification.py
```

The script limits numerical-library threads to two and does not use a GPU.
It emits a pre-execution plan, per-run NPZ/JSON, errors.csv, summary.json,
checksums.json, and convergence.png/pdf. The actual environment, git state,
script/protocol/production-solver hashes and measurements accompany each run.
Development in a reused Windows CPU environment must be labelled separately
from the final clean Linux execution. Run success does not imply byte-identical
floating-point arrays on every platform.

Chart contract: the question is whether errors decrease at the expected order
under the declared refinements; the takeaway is conditional on the observed
gates, not prewritten as a pass. Use static log-log line plots with markers,
three declared levels per series, axes labelled with grid cell count or time
step and physical error units, and reference-slope guides. Different marker
shapes distinguish the two steady cases as well as different colors. Export
PNG/PDF for the revised-paper package and inspect the PNG visually. Exact
metrics and any failed gates remain available in the CSV and JSON.

## Scope

These cases strengthen mathematical implementation verification for anisotropic
two-dimensional diffusion, steady Robin heat transfer, a aligned conductivity
interface, and a solvable special-case reaction/heat coupling. They do not
establish experimental validation, COMSOL agreement, validity of real material
parameters, general nonlinear temperature-dependent cure accuracy, curved
geometry, or nonuniform-grid order. Existing benchmark checks retain their own
scope and should not be relabelled independent external validation.

Methodological context: [NASA verification assessment](https://www.grc.nasa.gov/www/wind/valid/tutorial/verassess.html)
and [Salari and Knupp, Code Verification by the Method of Manufactured Solutions](https://www.osti.gov/biblio/759450/).
