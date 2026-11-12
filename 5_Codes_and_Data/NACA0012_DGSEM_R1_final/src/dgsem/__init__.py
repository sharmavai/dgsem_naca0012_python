"""dgsem -- A nodal discontinuous Galerkin spectral element solver for the
two-dimensional compressible Euler / Navier-Stokes equations on curvilinear
meshes, applied to flow past the NACA 0012 airfoil.

The discretisation follows the discontinuous Galerkin spectral element method
(DGSEM) with Legendre-Gauss-Lobatto (LGL) collocation points, conservative
(curl-form) metric terms that satisfy the discrete metric identities exactly,
and Rusanov or Roe interface fluxes.  Steady states are reached with explicit
strong-stability-preserving Runge-Kutta time stepping and local (per-element)
time steps.

Modules
-------
quadrature   LGL nodes, weights, differentiation and interpolation matrices.
geometry     NACA 0012 contour definition (sharp/open trailing edge, closed
             with a small straight base) and arc-length parameterisation.
mesh         Body-fitted O-type mesh generation, metric terms, quality checks.
euler        Conservative variables, primitive recovery, physical fluxes.
numflux      Rusanov and Roe (with Harten entropy fix) numerical fluxes.
operator     DGSEM spatial operator (volume, interface, boundary terms) and
             BR2/SIPG viscous terms with element-wise artificial viscosity.
shock        Persson-Peraire type modal shock sensor and artificial viscosity.
limiters     Zhang-Shu type positivity-preserving scaling limiter.
timestep     Local time-step estimate (inviscid and viscous restrictions).
integrate    SSP-RK3 / classical RK4 time stepping with local time steps.
forces       Surface pressure integration: Cp, Cl, Cd, Cm.
transfer     Solution transfer between nested meshes (parameter-space
             Lagrange interpolation) for nested initialisation.
solver       Case setup and steady-state driver with monitoring and restart.
sweep        Angle-of-attack sweep driver with solution continuation.
verification Verification suite (free-stream preservation, isentropic vortex
             advection order study, viscous-operator accuracy).
experimental Experimental reference-data loading.
output       Machine-readable result output (npz / CSV).
figures      Publication-quality figure generation (PNG / PDF / SVG + CSV).
"""

__version__ = "1.0.0"
