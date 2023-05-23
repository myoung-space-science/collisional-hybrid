/* 3D Hybrid PIC */
static char help[] = "A 3D hybrid particle-in-cell (PIC) simulation.";

#include <time.h>
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include <petscviewerhdf5.h>
#include "hybrid.h"
#include "random.h"
#include "lhs.h"
#include "rhs.h"
#include "output.h"

const char *RHSTypes[] = {
  "constant", "sinusoidal", "full", "RHSType", "RHS_", NULL
};

const char *LHSTypes[] = {
  "identity", "laplacian", "full", "LHSType", "LHS_", NULL
};

const char *DensityTypes[] ={
  "flat-sobol", "flat-reverse", "flat-normal", "uniform", "sinusoidal", "gaussian", "DensityType", "DENSITY_", NULL
};

const char *BCTypes[] = {
  "periodic", "injection", "reflection", "dirichlet", "neumann", "BCType", "BC_", NULL
};

static PetscErrorCode
ProcessOptions(Context *ctx)
{
  /* This could read all options from the command line, but we may want to use
  libconfig to read most of the options from a user-provided file, and only
  support a few runtime CLI options (e.g., output directory). Note that the
  third argument to PetscInitialize is the name of an options-database file, so
  it may be possible to specify all runtime options that way. */

  PetscFunctionBeginUser;

  PetscInt  intArg;
  PetscReal realArg;
  PetscBool boolArg;
  PetscEnum enumArg;
  PetscBool found;
  PetscReal tmp;

  // Set fundamental parameter values.
  ctx->electrons.q = -Q;
  ctx->electrons.m = ME;

  // Read optional parameter values from user input.
  PetscCall(PetscOptionsGetBool(NULL, NULL, "--view-lhs", &boolArg, &found));
  if (found) {
    ctx->viewLHS = boolArg;
  } else {
    ctx->viewLHS = PETSC_FALSE;
  }
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "--rhs-type", RHSTypes, &enumArg, &found));
  if (found) {
    ctx->rhsType = enumArg;
  } else {
    ctx->rhsType = RHS_FULL;
  }
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "--lhs-type", LHSTypes, &enumArg, &found));
  if (found) {
    ctx->lhsType = enumArg;
  } else {
    ctx->lhsType = LHS_FULL;
  }
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "--density-type", DensityTypes, &enumArg, &found));
  if (found) {
    ctx->densityType = enumArg;
  } else {
    ctx->densityType = DENSITY_FLAT_SOBOL;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Nx", &intArg, &found));
  if (found) {
    if (intArg < 0) {
      PRINT_WORLD("Warning: Ignoring negative value for Nx: %d\n", intArg);
    } else {
      ctx->grid.N.x = intArg;
    }
  } else {
    ctx->grid.N.x = -1;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Ny", &intArg, &found));
  if (found) {
    if (intArg < 0) {
      PRINT_WORLD("Warning: Ignoring negative value for Ny: %d\n", intArg);
    } else {
      ctx->grid.N.y = intArg;
    }
  } else {
    ctx->grid.N.y = -1;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Nz", &intArg, &found));
  if (found) {
    if (intArg < 0) {
      PRINT_WORLD("Warning: Ignoring negative value for Nz: %d\n", intArg);
    } else {
      ctx->grid.N.z = intArg;
    }
  } else {
    ctx->grid.N.z = -1;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-x0", &realArg, &found));
  if (found) {
    ctx->grid.p0.x = realArg;
  } else {
    ctx->grid.p0.x = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-y0", &realArg, &found));
  if (found) {
    ctx->grid.p0.y = realArg;
  } else {
    ctx->grid.p0.y = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-z0", &realArg, &found));
  if (found) {
    ctx->grid.p0.z = realArg;
  } else {
    ctx->grid.p0.z = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-x1", &realArg, &found));
  if (found) {
    ctx->grid.p1.x = realArg;
  } else {
    ctx->grid.p1.x = 1.0;
  }
  if (ctx->grid.p1.x == ctx->grid.p0.x) {
      PRINT_WORLD("Warning: zero-width x dimension\n");
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-y1", &realArg, &found));
  if (found) {
    ctx->grid.p1.y = realArg;
  } else {
    ctx->grid.p1.y = 1.0;
  }
  if (ctx->grid.p1.y == ctx->grid.p0.y) {
      PRINT_WORLD("Warning: zero-width y dimension\n");
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-z1", &realArg, &found));
  if (found) {
    ctx->grid.p1.z = realArg;
  } else {
    ctx->grid.p1.z = 1.0;
  }
  if (ctx->grid.p1.z == ctx->grid.p0.z) {
      PRINT_WORLD("Warning: zero-width z dimension\n");
  }
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "--x0-bc", BCTypes, &enumArg, &found));
  if (found) {
    ctx->xBC[0] = enumArg;
  } else {
    ctx->xBC[0] = BC_PERIODIC;
  }
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "--x1-bc", BCTypes, &enumArg, &found));
  if (found) {
    ctx->xBC[1] = enumArg;
  } else {
    ctx->xBC[1] = BC_PERIODIC;
  }
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "--y0-bc", BCTypes, &enumArg, &found));
  if (found) {
    ctx->yBC[0] = enumArg;
  } else {
    ctx->yBC[0] = BC_PERIODIC;
  }
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "--y1-bc", BCTypes, &enumArg, &found));
  if (found) {
    ctx->yBC[1] = enumArg;
  } else {
    ctx->yBC[1] = BC_PERIODIC;
  }
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "--z0-bc", BCTypes, &enumArg, &found));
  if (found) {
    ctx->zBC[0] = enumArg;
  } else {
    ctx->zBC[0] = BC_PERIODIC;
  }
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "--z1-bc", BCTypes, &enumArg, &found));
  if (found) {
    ctx->zBC[1] = enumArg;
  } else {
    ctx->zBC[1] = BC_PERIODIC;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Np", &intArg, &found));
  if (found) {
    ctx->plasma.Np = intArg;
  } else {
    ctx->plasma.Np = -1;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mn", &realArg, &found));
  if (found) {
    ctx->neutrals.m = realArg;
  } else {
    ctx->neutrals.m = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Tn", &realArg, &found));
  if (found) {
    ctx->neutrals.T = realArg;
  } else {
    ctx->neutrals.T = -1.0;
  }
  // The user may provide a single neutral-species drift speed for all
  // components, as well as a drift speed for each component. Any component
  // value not explicitly set will have the common value, which defaults to 0.
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vn0", &realArg, &found));
  if (found) {
    ctx->neutrals.v0.x = realArg;
    ctx->neutrals.v0.y = realArg;
    ctx->neutrals.v0.z = realArg;
  } else {
    ctx->neutrals.v0.x = 0.0;
    ctx->neutrals.v0.y = 0.0;
    ctx->neutrals.v0.z = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vn0x", &realArg, &found));
  if (found) {
    ctx->neutrals.v0.x = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vn0y", &realArg, &found));
  if (found) {
    ctx->neutrals.v0.y = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vn0z", &realArg, &found));
  if (found) {
    ctx->neutrals.v0.z = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-qi", &realArg, &found));
  if (found) {
    ctx->ions.q = realArg;
  } else {
    ctx->ions.q = Q;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mi", &realArg, &found));
  if (found) {
    ctx->ions.m = realArg;
  } else {
    ctx->ions.m = MP;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-nue", &realArg, &found));
  if (found) {
    ctx->electrons.nu = realArg;
  } else {
    ctx->electrons.nu = 1.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-nui", &realArg, &found));
  if (found) {
    ctx->ions.nu = realArg;
  } else {
    ctx->ions.nu = 1.0;
  }
  // The user may provide a single electron drift speed for all components, as
  // well as a drift speed for each component. Any component value not
  // explicitly set will have the common value, which defaults to 0.
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ve0", &realArg, &found));
  if (found) {
    ctx->electrons.v0.x = realArg;
    ctx->electrons.v0.y = realArg;
    ctx->electrons.v0.z = realArg;
  } else {
    ctx->electrons.v0.x = 0.0;
    ctx->electrons.v0.y = 0.0;
    ctx->electrons.v0.z = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ve0x", &realArg, &found));
  if (found) {
    ctx->electrons.v0.x = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ve0y", &realArg, &found));
  if (found) {
    ctx->electrons.v0.y = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ve0z", &realArg, &found));
  if (found) {
    ctx->electrons.v0.z = realArg;
  }
  // The user may provide a single ion drift speed for all components, as well
  // as a drift speed for each component. Any component value not explicitly set
  // will have the common value, which defaults to 0.
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vi0", &realArg, &found));
  if (found) {
    ctx->ions.v0.x = realArg;
    ctx->ions.v0.y = realArg;
    ctx->ions.v0.z = realArg;
  } else {
    ctx->ions.v0.x = 0.0;
    ctx->ions.v0.y = 0.0;
    ctx->ions.v0.z = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vi0x", &realArg, &found));
  if (found) {
    ctx->ions.v0.x = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vi0y", &realArg, &found));
  if (found) {
    ctx->ions.v0.y = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vi0z", &realArg, &found));
  if (found) {
    ctx->ions.v0.z = realArg;
  }
  // The user may provide a single electron thermal speed for all components, as
  // well as a thermal speed for each component. Any component value not
  // explicitly set will have the common value, which defaults to 0.
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-veT", &realArg, &found));
  if (found) {
    ctx->electrons.vT.x = realArg;
    ctx->electrons.vT.y = realArg;
    ctx->electrons.vT.z = realArg;
  } else {
    ctx->electrons.vT.x = 0.0;
    ctx->electrons.vT.y = 0.0;
    ctx->electrons.vT.z = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-veTx", &realArg, &found));
  if (found) {
    ctx->electrons.vT.x = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-veTy", &realArg, &found));
  if (found) {
    ctx->electrons.vT.y = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-veTz", &realArg, &found));
  if (found) {
    ctx->electrons.vT.z = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Te", &realArg, &found));
  if (found) {
    ctx->electrons.T = realArg;
  } else {
    ctx->electrons.T = 0.0;
  }
  // The user may provide a single ion thermal speed for all components, as well
  // as a thermal speed for each component. Any component value not explicitly
  // set will have the common value, which defaults to 0.
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-viT", &realArg, &found));
  if (found) {
    ctx->ions.vT.x = realArg;
    ctx->ions.vT.y = realArg;
    ctx->ions.vT.z = realArg;
  } else {
    ctx->ions.vT.x = 0.0;
    ctx->ions.vT.y = 0.0;
    ctx->ions.vT.z = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-viTx", &realArg, &found));
  if (found) {
    ctx->ions.vT.x = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-viTy", &realArg, &found));
  if (found) {
    ctx->ions.vT.y = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-viTz", &realArg, &found));
  if (found) {
    ctx->ions.vT.z = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Ti", &realArg, &found));
  if (found) {
    ctx->ions.T = realArg;
  } else {
    ctx->ions.T = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-B0x", &realArg, &found));
  if (found) {
    ctx->plasma.B0.x = realArg;
  } else {
    ctx->plasma.B0.x = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-B0y", &realArg, &found));
  if (found) {
    ctx->plasma.B0.y = realArg;
  } else {
    ctx->plasma.B0.y = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-B0z", &realArg, &found));
  if (found) {
    ctx->plasma.B0.z = realArg;
  } else {
    ctx->plasma.B0.z = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-E0x", &realArg, &found));
  if (found) {
    ctx->plasma.E0.x = realArg;
  } else {
    ctx->plasma.E0.x = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-E0y", &realArg, &found));
  if (found) {
    ctx->plasma.E0.y = realArg;
  } else {
    ctx->plasma.E0.y = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-E0z", &realArg, &found));
  if (found) {
    ctx->plasma.E0.z = realArg;
  } else {
    ctx->plasma.E0.z = 0.0;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Nt", &intArg, &found));
  if (found) {
    ctx->Nt = intArg;
  } else {
    ctx->Nt = 1;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-dt", &realArg, &found));
  if (found) {
    ctx->dt = realArg;
  } else {
    PRINT_WORLD("Warning: Setting dt = 1 / nui\n");
    ctx->dt = 1.0 / ctx->ions.nu;
  }

  // Set the LHS function based on LHS type.
  switch (ctx->lhsType) {
  case LHS_IDENTITY:
    ctx->lhsFunc = ComputeIdentityLHS;
    break;
  case LHS_LAPLACIAN:
    ctx->lhsFunc = ComputeLaplacianLHS;
    break;
  case LHS_FULL:
    ctx->lhsFunc = ComputeFullLHS;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Unknown LHS type: \"%s\"\n", LHSTypes[ctx->lhsType]);
  }
  // Set RHS function based on RHS type.
  switch (ctx->rhsType) {
  case RHS_CONSTANT:
    ctx->rhsFunc = ComputeConstantRHS;
    break;
  case RHS_SINUSOIDAL:
    ctx->rhsFunc = ComputeSinusoidalRHS;
    break;
  case RHS_FULL:
    ctx->rhsFunc = ComputeFullRHS;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Unknown RHS type: \"%s\"\n", RHSTypes[ctx->rhsType]);
  }
  // Set grid lengths from lower and upper bounds.
  ctx->grid.L.x = ctx->grid.p1.x - ctx->grid.p0.x;
  ctx->grid.L.y = ctx->grid.p1.y - ctx->grid.p0.y;
  ctx->grid.L.z = ctx->grid.p1.z - ctx->grid.p0.z;
  // Set up boundary conditions. If one is periodic, both must be periodic.
  if ((ctx->xBC[0] == BC_PERIODIC) && (ctx->xBC[1] == BC_PERIODIC)) {
    ctx->xDMBC = DM_BOUNDARY_PERIODIC;
  } else if ((ctx->xBC[0] != BC_PERIODIC) && (ctx->xBC[1] != BC_PERIODIC)) {
    ctx->xDMBC = DM_BOUNDARY_GHOSTED;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Inconsistent x-axis boundary conditions: {%s, %s}\n", BCTypes[ctx->xBC[0]], BCTypes[ctx->xBC[1]]);
  }
  if ((ctx->yBC[0] == BC_PERIODIC) && (ctx->yBC[1] == BC_PERIODIC)) {
    ctx->yDMBC = DM_BOUNDARY_PERIODIC;
  } else if ((ctx->yBC[0] != BC_PERIODIC) && (ctx->yBC[1] != BC_PERIODIC)) {
    ctx->yDMBC = DM_BOUNDARY_GHOSTED;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Inconsistent y-axis boundary conditions: {%s, %s}\n", BCTypes[ctx->yBC[0]], BCTypes[ctx->yBC[1]]);
  }
  if ((ctx->zBC[0] == BC_PERIODIC) && (ctx->zBC[1] == BC_PERIODIC)) {
    ctx->zDMBC = DM_BOUNDARY_PERIODIC;
  } else if ((ctx->zBC[0] != BC_PERIODIC) && (ctx->zBC[1] != BC_PERIODIC)) {
    ctx->zDMBC = DM_BOUNDARY_GHOSTED;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Inconsistent z-axis boundary conditions: {%s, %s}\n", BCTypes[ctx->zBC[0]], BCTypes[ctx->zBC[1]]);
  }
  // Set species gyrofrequency from q, B0, and m.
  ctx->electrons.Omega.x = PetscAbsReal(ctx->electrons.q * ctx->plasma.B0.x / ctx->electrons.m);
  ctx->electrons.Omega.y = PetscAbsReal(ctx->electrons.q * ctx->plasma.B0.y / ctx->electrons.m);
  ctx->electrons.Omega.z = PetscAbsReal(ctx->electrons.q * ctx->plasma.B0.z / ctx->electrons.m);
  ctx->ions.Omega.x = PetscAbsReal(ctx->ions.q * ctx->plasma.B0.x / ctx->ions.m);
  ctx->ions.Omega.y = PetscAbsReal(ctx->ions.q * ctx->plasma.B0.y / ctx->ions.m);
  ctx->ions.Omega.z = PetscAbsReal(ctx->ions.q * ctx->plasma.B0.z / ctx->ions.m);
  // Set species magnetization from Omega and nu.
  ctx->electrons.kappa.x = ctx->electrons.Omega.x / ctx->electrons.nu;
  ctx->electrons.kappa.y = ctx->electrons.Omega.y / ctx->electrons.nu;
  ctx->electrons.kappa.z = ctx->electrons.Omega.z / ctx->electrons.nu;
  ctx->ions.kappa.x = ctx->ions.Omega.x / ctx->ions.nu;
  ctx->ions.kappa.y = ctx->ions.Omega.y / ctx->ions.nu;
  ctx->ions.kappa.z = ctx->ions.Omega.z / ctx->ions.nu;
  // Compute drift-velocity magnitudes.
  ctx->electrons.v0.r = PetscSqrtReal(PetscSqr(ctx->electrons.v0.x) + PetscSqr(ctx->electrons.v0.y) + PetscSqr(ctx->electrons.v0.z));
  ctx->ions.v0.r = PetscSqrtReal(PetscSqr(ctx->ions.v0.x) + PetscSqr(ctx->ions.v0.y) + PetscSqr(ctx->ions.v0.z));
  ctx->neutrals.v0.r = PetscSqrtReal(PetscSqr(ctx->neutrals.v0.x) + PetscSqr(ctx->neutrals.v0.y) + PetscSqr(ctx->neutrals.v0.z));
  // Make electron temperature and thermal velocity consistent.
  if ((ctx->electrons.vT.x != 0.0) || (ctx->electrons.vT.y != 0.0) || (ctx->electrons.vT.z != 0.0)) {
    ctx->electrons.vT.r = PetscSqrtReal(PetscSqr(ctx->electrons.vT.x) + PetscSqr(ctx->electrons.vT.y) + PetscSqr(ctx->electrons.vT.z));
    ctx->electrons.T = (0.5 * ctx->electrons.m / KB) * (PetscSqr(ctx->electrons.vT.r));
  } else {
    tmp = PetscSqrtReal(2.0 * KB * ctx->electrons.T / ctx->electrons.m) / 3.0;
    ctx->electrons.vT.x = tmp;
    ctx->electrons.vT.y = tmp;
    ctx->electrons.vT.z = tmp;
    ctx->electrons.vT.r = PetscSqrtReal(PetscSqr(ctx->electrons.vT.x) + PetscSqr(ctx->electrons.vT.y) + PetscSqr(ctx->electrons.vT.z));
  }
  // Make ion temperature and thermal velocity consistent.
  if ((ctx->ions.vT.x != 0.0) || (ctx->ions.vT.y != 0.0) || (ctx->ions.vT.z != 0.0)) {
    ctx->ions.vT.r = PetscSqrtReal(PetscSqr(ctx->ions.vT.x) + PetscSqr(ctx->ions.vT.y) + PetscSqr(ctx->ions.vT.z));
    ctx->ions.T = (0.5 * ctx->ions.m / KB) * (PetscSqr(ctx->ions.vT.r));
  } else {
    tmp = PetscSqrtReal(2.0 * KB * ctx->ions.T / ctx->ions.m) / 3.0;
    ctx->ions.vT.x = tmp;
    ctx->ions.vT.y = tmp;
    ctx->ions.vT.z = tmp;
    ctx->ions.vT.r = PetscSqrtReal(PetscSqr(ctx->ions.vT.x) + PetscSqr(ctx->ions.vT.y) + PetscSqr(ctx->ions.vT.z));
  }
  // Set default neutral temperature based on charge species.
  if (ctx->neutrals.T == -1.0) {
    ctx->neutrals.T = ctx->ions.T;
    PRINT_WORLD("Warning: Setting neutral temperature equal to ion temperature (%.1f K)\n", ctx->ions.T);
  }
  // Set neutral thermal velocity from temperature.
  ctx->neutrals.vT = PetscSqrtReal(2.0 * KB * ctx->neutrals.T / ctx->neutrals.m);

  // TODO: Should we set default collision frequencies based on an analytic
  // formulation (e.g., from Schunk & Nagy)?

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeGridDM(DM *grid, Context *ctx)
{
  PetscInt       Nx=(ctx->grid.N.x > 0 ? ctx->grid.N.x : 7);
  PetscInt       Ny=(ctx->grid.N.y > 0 ? ctx->grid.N.y : 7);
  PetscInt       Nz=(ctx->grid.N.z > 0 ? ctx->grid.N.z : 7);
  DMBoundaryType xBC=ctx->xDMBC;
  DMBoundaryType yBC=ctx->yDMBC;
  DMBoundaryType zBC=ctx->zDMBC;
  PetscReal      dx, dy, dz;
  PetscInt       dof=4;
  PetscInt       width=1;

  PetscFunctionBeginUser;

  // Create the grid DM.
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, xBC, yBC, zBC, DMDA_STENCIL_BOX, Nx, Ny, Nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, width, NULL, NULL, NULL, grid));
  // Perform basic setup.
  PetscCall(DMDASetElementType(*grid, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(*grid));
  PetscCall(DMSetUp(*grid));
  // Synchronize values of Nx, Ny, and Nz passed via -n{x,y,z} or
  // -da_grid_{x,y,z}. Note that this gives precedence to the latter.
  PetscCall(DMDAGetInfo(*grid, NULL, &Nx, &Ny, &Nz, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  if (ctx->grid.N.x == -1) {
    ctx->grid.N.x = Nx;
  }
  if (ctx->grid.N.y == -1) {
    ctx->grid.N.y = Ny;
  }
  if (ctx->grid.N.z == -1) {
    ctx->grid.N.z = Nz;
  }
  // Set the number of charged particles equal to the number of grid cells, if
  // necessary. Note that this must occur after synchronizing grid cells.
  if (ctx->plasma.Np == -1) {
    ctx->plasma.Np = ctx->grid.N.x * ctx->grid.N.y * ctx->grid.N.z;
  }
  // Define the physical grid-cell spacing.
  dx = ctx->grid.L.x / (PetscReal)ctx->grid.N.x;
  dy = ctx->grid.L.y / (PetscReal)ctx->grid.N.y;
  dz = ctx->grid.L.z / (PetscReal)ctx->grid.N.z;
  ctx->grid.d.x = dx;
  ctx->grid.d.y = dy;
  ctx->grid.d.z = dz;
  // Set uniform coordinates on the grid DM.
  PetscCall(DMDASetUniformCoordinates(*grid, ctx->grid.p0.x, ctx->grid.p1.x+dx, ctx->grid.p0.y, ctx->grid.p1.y+dy, ctx->grid.p0.z, ctx->grid.p1.z+dz));
  // Declare grid-quantity names.
  PetscCall(DMDASetFieldName(*grid, 0, "density"));
  PetscCall(DMDASetFieldName(*grid, 1, "x flux"));
  PetscCall(DMDASetFieldName(*grid, 2, "y flux"));
  PetscCall(DMDASetFieldName(*grid, 3, "z flux"));
  // Associate the user context with the grid DM.
  PetscCall(DMSetApplicationContext(*grid, &ctx));
  // View information about the grid DM.
  PetscCall(DMView(*grid, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializePotentialDM(DM grid, DM *solve)
{
  PetscFunctionBeginUser;

  PetscCall(DMDACreateCompatibleDMDA(grid, 1, solve));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeSwarmDM(DM grid, Context *ctx)
{
  DM       swarm;
  PetscInt dim;
  PetscInt bufsize=0;
  PetscInt Np, np;

  PetscFunctionBeginUser;

  // Create the swarm DM.
  PetscCall(DMCreate(PETSC_COMM_WORLD, &swarm));
  // Perform basic setup.
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(swarm), "ions_"));
  PetscCall(DMSetType(swarm, DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject)swarm, "Ions"));
  // Synchronize the swarm DM with the grid DM.
  PetscCall(DMGetDimension(grid, &dim));
  PetscCall(DMSetDimension(swarm, dim));
  PetscCall(DMSwarmSetCellDM(swarm, grid));
  // Declare this to be a PIC swarm. This must occur after setting `dim`.
  PetscCall(DMSwarmSetType(swarm, DMSWARM_PIC));
  // Register fields that each particle will have.
  PetscCall(DMSwarmInitializeFieldRegister(swarm));
  // --> (x, y, z) position components
  PetscCall(DMSwarmRegisterUserStructField(swarm, "position", sizeof(RealVector)));
  // --> (x, y, z) velocity components
  PetscCall(DMSwarmRegisterUserStructField(swarm, "velocity", sizeof(RealVector)));
  PetscCall(DMSwarmFinalizeFieldRegister(swarm));
  // Set the per-processor swarm size and buffer length for efficient resizing.
  Np = (ctx->plasma.Np > 0) ? ctx->plasma.Np : NPTOTAL;
  np = (PetscInt)(Np / ctx->mpi.size);
  bufsize = (PetscInt)(0.25 * np);
  PetscCall(DMSwarmSetLocalSizes(swarm, np, bufsize));
  // View information about the swarm DM.
  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));
  // Assign the swarm to the application context.
  ctx->swarm = swarm;

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
UniformDistribution_FromSwarm(Context *ctx)
{
  DM          swarm=ctx->swarm;
  PetscReal   min[NDIM], max[NDIM];
  PetscInt    npoints[NDIM];

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  min[0] = ctx->grid.p0.x;
  min[1] = ctx->grid.p0.y;
  min[2] = ctx->grid.p0.z;
  max[0] = ctx->grid.p1.x - ctx->grid.d.x;
  max[1] = ctx->grid.p1.y - ctx->grid.d.y;
  max[2] = ctx->grid.p1.z - ctx->grid.d.z;
  npoints[0] = 4 * ctx->grid.N.x;
  npoints[1] = 4 * ctx->grid.N.y;
  npoints[2] = 4 * ctx->grid.N.z;

  PetscCall(DMSwarmSetPointsUniformCoordinates(swarm, min, max, npoints, INSERT_VALUES));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
UniformDistribution(Context *ctx)
{
  DM          swarm=ctx->swarm;
  PetscScalar *coords;
  PetscInt    np, np_cell, ip;
  DM          grid;
  PetscInt    i0, j0, k0;
  PetscInt    ni, nj, nk, nc;
  PetscInt    i, j, k, idx;
  PetscInt    dim;
  PetscReal   r[NDIM];
  PetscReal   d[NDIM]={ctx->grid.d.x, ctx->grid.d.y, ctx->grid.d.z};

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get information about the grid.
  PetscCall(DMSwarmGetCellDM(swarm, &grid));
  PetscCall(DMDAGetCorners(grid, &i0, &j0, &k0, &ni, &nj, &nk));

  // Get the local number of particles.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  // Compute the number of particles per cell. Note that np_cell*nc will
  // not in general be equal to the input value of -Np, if given.
  nc = ni*nj*nk;
  np_cell = (PetscInt)(np / nc);

  // Reset the local swarm size to avoid a seg fault when accessing the
  // coordinates array. Passing a negative value for the buffer forces the swarm
  // to use its existing buffer size.
  PetscCall(DMSwarmSetLocalSizes(swarm, np_cell*nc, -1));

  // Get a representation of the particle coordinates.
  PetscCall(DMSwarmGetField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Loop over cells and place an equal number of particles at the center of
  // each cell in all but the last row of each index. This will result in a
  // symmetric step-function distribution. It is not truly uniform.
  for (ip=0; ip<np_cell; ip++) {
    for (i=i0; i<i0+ni-1; i++) {
      for (j=j0; j<j0+nj-1; j++) {
        for (k=k0; k<k0+nk-1; k++) {
          idx = (ip*nc + k + j*nk + i*nk*nj)*NDIM;
          r[0] = (PetscReal)(i + 1);
          r[1] = (PetscReal)(j + 1);
          r[2] = (PetscReal)(k + 1);
          for (dim=0; dim<NDIM; dim++) {
            coords[idx + dim] = d[dim]*(r[dim] - 0.5);
          }
        }
      }
    }
  }

  // Restore the coordinates array.
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
SobolDistribution(Context *ctx)
{
  DM          swarm=ctx->swarm;
  PetscInt    seed=-1, ndim=NDIM;
  PetscReal   *coords, v;
  PetscInt    np, ip;
  PetscReal   r[NDIM];
  PetscReal   L[NDIM]={ctx->grid.L.x, ctx->grid.L.y, ctx->grid.L.z};
  PetscReal   d[NDIM]={ctx->grid.d.x, ctx->grid.d.y, ctx->grid.d.z};
  PetscReal   p0[NDIM]={ctx->grid.p0.x, ctx->grid.p0.y, ctx->grid.p0.z};
  PetscReal   p1[NDIM]={ctx->grid.p1.x, ctx->grid.p1.y, ctx->grid.p1.z};
  PetscInt    dim;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get a representation of the particle coordinates.
  PetscCall(DMSwarmGetField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Initialize the psuedo-random number generator.
  PetscCall(Sobseq(&seed, r-1));

  // Get the local number of particles.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  for (ip=0; ip<np; ip++) {
    PetscCall(Sobseq(&ndim, r-1));
    for (dim=0; dim<NDIM; dim++) {
      v = r[dim]*(L[dim] + d[dim]) - 0.5*d[dim];
      if (v < p0[dim]) {
        coords[ip*NDIM + dim] = p0[dim] + 0.01*d[dim];
      } else if (v > p1[dim]) {
        coords[ip*NDIM + dim] = p1[dim] - 0.01*d[dim];
      } else {
        coords[ip*NDIM + dim] = v;
      }
    }
  }

  // Restore the coordinates array.
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


// Type to be used for particle distribution functions.
typedef PetscErrorCode (*DistributionFunction)(PetscReal x, PetscReal y, PetscReal z, PetscReal *v, Context *ctx);


PetscErrorCode
SinusoidalDistribution(PetscReal x, PetscReal y, PetscReal z, PetscReal *v, Context *ctx)
{
  PetscReal fx;

  PetscFunctionBeginUser;

  fx = PetscSinReal(2*PETSC_PI * x);
  *v = 1.0 + 0.25*fx;

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
Rejection(DistributionFunction density, Context *ctx)
{
  PetscRandom random;
  DM          swarm=ctx->swarm;
  PetscInt    np, ip;
  DM          grid;
  PetscInt    i0, ni, i;
  PetscInt    j0, nj, j;
  PetscInt    k0, nk, k;
  PetscReal   localMax=0.0;
  PetscScalar *coords;
  PetscReal   Lx=ctx->grid.L.x;
  PetscReal   Ly=ctx->grid.L.y;
  PetscReal   Lz=ctx->grid.L.z;
  PetscReal   x, y, z, v, w;
  PetscReal   r[NDIM];
  PetscInt    it=0;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get a representation of the particle coordinates.
  PetscCall(DMSwarmGetField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Create a random number generator.
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &random));
  PetscCall(PetscRandomSetInterval(random, 0.0, 1.0));
  PetscCall(PetscRandomSetSeed(random, ctx->mpi.rank));
  PetscCall(PetscRandomSeed(random));

  // Compute local maximum density.
  PetscCall(DMSwarmGetCellDM(swarm, &grid));
  PetscCall(DMDAGetCorners(grid, &i0, &j0, &k0, &ni, &nj, &nk));
  for (i=i0; i<i0+ni; i++) {
    for (j=j0; j<j0+nj; j++) {
      for (k=k0; k<k0+nk; k++) {
        PetscCall(density(i, j, k, &w, ctx));
        localMax = PetscMax(localMax, w);
      }
    }
  }
  PRINT_RANKS("[%d] Local maximum density: %g\n", ctx->mpi.rank, localMax);

  // Get the local number of particles.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  // Loop over all local particles.
  ip = 0;
  while (ip < np) {
    PetscCall(PetscRandomGetValuesReal(random, NDIM, r));
    x = r[0] * Lx;
    y = r[1] * Ly;
    z = r[2] * Lz;
    PetscCall(density(x, y, z, &w, ctx));
    PetscCall(PetscRandomGetValueReal(random, &v));
    if (w > v * localMax) {
      coords[ip*NDIM + 0] = x;
      coords[ip*NDIM + 1] = y;
      coords[ip*NDIM + 2] = z;
      ip++;
    }
    it++;
  }

  // Echo rejection efficiency.
  NEWLINE;
  PRINT_RANKS("[%d] Rejection efficiency: %f\n", ctx->mpi.rank, (PetscReal)ip/it);

  // Restore the coordinates array.
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Destroy the random-number generator.
  PetscCall(PetscRandomDestroy(&random));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializePositions(Context *ctx)
{
  DM          swarm=ctx->swarm;
  PetscInt    np, Np, ip;
  PetscScalar *coords;
  RealVector  *pos;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Echo sizes.
  PetscCall(DMSwarmGetSize(swarm, &Np));
  PetscCall(DMSwarmGetLocalSize(swarm, &np));
  NEWLINE;
  PRINT_RANKS("[%d] Local # of particles before placement: %d\n", ctx->mpi.rank, np);
  PRINT_WORLD("   Global # of particles before placement: %d\n", Np);

  // Initialize coordinates in the particle DM.
  switch(ctx->densityType) {
    case DENSITY_FLAT_NORMAL:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Not implemented: %s density", DensityTypes[DENSITY_FLAT_NORMAL]);
      break;
    case DENSITY_FLAT_REVERSE:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Not implemented: %s density", DensityTypes[DENSITY_FLAT_REVERSE]);
      break;
    case DENSITY_FLAT_SOBOL:
      PetscCall(SobolDistribution(ctx));
      break;
    case DENSITY_UNIFORM:
      PetscCall(UniformDistribution(ctx));
      break;
    case DENSITY_SINUSOIDAL:
      PetscCall(Rejection(SinusoidalDistribution, ctx));
      break;
    case DENSITY_GAUSSIAN:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Not implemented: %s density", DensityTypes[DENSITY_GAUSSIAN]);
      break;
  }

  // Echo sizes.
  PetscCall(DMSwarmGetSize(swarm, &Np));
  PetscCall(DMSwarmGetLocalSize(swarm, &np));
  NEWLINE;
  PRINT_RANKS("[%d] Local # of particles after placement: %d\n", ctx->mpi.rank, np);
  PRINT_WORLD("   Global # of particles after placement: %d\n", Np);

  // Migrate particles between ranks.
  PetscCall(DMSwarmMigrate(swarm, PETSC_TRUE));

  // Echo sizes.
  PetscCall(DMSwarmGetSize(swarm, &Np));
  PetscCall(DMSwarmGetLocalSize(swarm, &np));
  NEWLINE;
  PRINT_RANKS("[%d] Local # of particles after migration: %d\n", ctx->mpi.rank, np);
  PRINT_WORLD("   Global # of particles after migration: %d\n", Np);

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  // Get an array representation of the swarm coordinates.
  PetscCall(DMSwarmGetField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Get an array representation of the particle positions.
  PetscCall(DMSwarmGetField(swarm, "position", NULL, NULL, (void **)&pos));

  // Loop over particles and assign parameter values.
  for (ip=0; ip<np; ip++) {
    pos[ip].x = coords[ip*NDIM + 0];
    pos[ip].y = coords[ip*NDIM + 1];
    pos[ip].z = coords[ip*NDIM + 2];
  }

  // Restore the particle-positions array.
  PetscCall(DMSwarmRestoreField(swarm, "position", NULL, NULL, (void **)&pos));

  // Restore the swarm-coordinates array.
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Display information about the particle DM.
  NEWLINE;
  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));

  // Update the parameter context.
  PetscCall(DMSwarmGetSize(swarm, &ctx->plasma.Np));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeVelocities(Context *ctx)
{
  DM          swarm=ctx->swarm;
  PetscInt    np, ip;
  RealVector  *vel;
  PetscReal   dvx, dvy, dvz;
  long        seed=ctx->seed;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get the number of local particles.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  // Get an array representation of the particle velocities.
  PetscCall(DMSwarmGetField(swarm, "velocity", NULL, NULL, (void **)&vel));

  // Loop over particles and assign parameter values.
  for (ip=0; ip<np; ip++) {
    PetscCall(Gasdev(&seed, &dvx));
    PetscCall(Gasdev(&seed, &dvy));
    PetscCall(Gasdev(&seed, &dvz));
    vel[ip].x = ctx->ions.vT.x*dvx + ctx->ions.v0.x;
    vel[ip].y = ctx->ions.vT.y*dvy + ctx->ions.v0.y;
    vel[ip].z = ctx->ions.vT.z*dvz + ctx->ions.v0.z;
  }

  // Restore the particle-velocities array.
  PetscCall(DMSwarmRestoreField(swarm, "velocity", NULL, NULL, (void **)&vel));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
EchoSetup(Context ctx)
{
  PetscViewer viewer;

  PetscFunctionBeginUser;

  // Open the viewer.
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "options.txt", &viewer));

  // View the PETSc options database.
  PetscCall(PetscOptionsView(NULL, viewer));

  // View custom parameter values.
  PetscCall(PetscViewerASCIIPrintf(viewer, "\nParameter Values\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer,   "----------------\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Nt = %d\n", ctx.Nt));
  PetscCall(PetscViewerASCIIPrintf(viewer, "dt = %f [s]\n", ctx.dt));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Nx = %d\n", ctx.grid.N.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Ny = %d\n", ctx.grid.N.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Nz = %d\n", ctx.grid.N.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "x0 = %f [m]\n", ctx.grid.p0.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "y0 = %f [m]\n", ctx.grid.p0.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "z0 = %f [m]\n", ctx.grid.p0.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "x1 = %f [m]\n", ctx.grid.p1.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "y1 = %f [m]\n", ctx.grid.p1.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "z1 = %f [m]\n", ctx.grid.p1.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Lx = %f [m]\n", ctx.grid.L.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Ly = %f [m]\n", ctx.grid.L.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Lz = %f [m]\n", ctx.grid.L.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "dx = %f [m]\n", ctx.grid.d.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "dy = %f [m]\n", ctx.grid.d.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "dz = %f [m]\n", ctx.grid.d.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Np = %d\n", ctx.plasma.Np));
  PetscCall(PetscViewerASCIIPrintf(viewer, "B0x = %g [T]\n", ctx.plasma.B0.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "B0y = %g [T]\n", ctx.plasma.B0.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "B0z = %g [T]\n", ctx.plasma.B0.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "E0x = %g [N/C]\n", ctx.plasma.E0.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "E0y = %g [N/C]\n", ctx.plasma.E0.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "E0z = %g [N/C]\n", ctx.plasma.E0.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "nue = %g [s^-1]\n", ctx.electrons.nu));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ve0x = %f [m/s]\n", ctx.electrons.v0.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ve0y = %f [m/s]\n", ctx.electrons.v0.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ve0z = %f [m/s]\n", ctx.electrons.v0.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "veTx = %f [m/s]\n", ctx.electrons.vT.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "veTy = %f [m/s]\n", ctx.electrons.vT.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "veTz = %f [m/s]\n", ctx.electrons.vT.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Te = %f [K]\n", ctx.electrons.T));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Omega_ex = %g [s^-1]\n", ctx.electrons.Omega.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Omega_ey = %g [s^-1]\n", ctx.electrons.Omega.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Omega_ez = %g [s^-1]\n", ctx.electrons.Omega.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "kappa_ex = %g\n", ctx.electrons.kappa.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "kappa_ey = %g\n", ctx.electrons.kappa.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "kappa_ez = %g\n", ctx.electrons.kappa.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "qi = %e [C]\n", ctx.ions.q));
  PetscCall(PetscViewerASCIIPrintf(viewer, "mi = %e [kg]\n", ctx.ions.m));
  PetscCall(PetscViewerASCIIPrintf(viewer, "nui = %g [s^-1]\n", ctx.ions.nu));
  PetscCall(PetscViewerASCIIPrintf(viewer, "vi0x = %f [m/s]\n", ctx.ions.v0.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "vi0y = %f [m/s]\n", ctx.ions.v0.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "vi0z = %f [m/s]\n", ctx.ions.v0.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "viTx = %f [m/s]\n", ctx.ions.vT.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "viTy = %f [m/s]\n", ctx.ions.vT.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "viTz = %f [m/s]\n", ctx.ions.vT.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Ti = %f [K]\n", ctx.ions.T));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Omega_ix = %g [s^-1]\n", ctx.ions.Omega.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Omega_iy = %g [s^-1]\n", ctx.ions.Omega.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Omega_iz = %g [s^-1]\n", ctx.ions.Omega.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "kappa_ix = %g\n", ctx.ions.kappa.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "kappa_iy = %g\n", ctx.ions.kappa.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "kappa_iz = %g\n", ctx.ions.kappa.z));
  PetscCall(PetscViewerASCIIPrintf(viewer, "mn = %e\n", ctx.neutrals.m));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Tn = %f\n", ctx.neutrals.T));
  PetscCall(PetscViewerASCIIPrintf(viewer, "vnT = %f\n", ctx.neutrals.vT));
  PetscCall(PetscViewerASCIIPrintf(viewer, "vn0x = %f\n", ctx.neutrals.v0.x));
  PetscCall(PetscViewerASCIIPrintf(viewer, "vn0y = %f\n", ctx.neutrals.v0.y));
  PetscCall(PetscViewerASCIIPrintf(viewer, "vn0z = %f\n", ctx.neutrals.v0.z));

  // Free memory.
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CollectParticles(Context *ctx)
{
  DM          grid, swarm=ctx->swarm;
  Vec         gridvec;
  GridNode    ***array;
  PetscInt    dim;
  PetscReal   x, y, z, dx, dy, dz;
  RealVector  r, *pos, v, *vel;
  PetscInt    ip, np;
  PetscInt    ixl, ixh, iyl, iyh, izl, izh;
  PetscReal   wxl, wxh, wyl, wyh, wzl, wzh;
  PetscReal   hhh, lhh, hlh, llh, hhl, lhl, hll, lll;
  PetscReal   w[NDIM];

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get the grid DM from the swarm DM.
  PetscCall(DMSwarmGetCellDM(swarm, &grid));

  // Get a vector for the local portion of the grid.
  PetscCall(DMGetLocalVector(grid, &gridvec));

  // Make sure the local grid vector has zeroes everywhere.
  PetscCall(VecZeroEntries(gridvec));

  // Get an array corresponding to the local grid quantities. Creating this
  // array from the local vector causes it to have room for the ghost points.
  PetscCall(DMDAVecGetArray(grid, gridvec, &array));

  // Get an array representation of the ion positions.
  PetscCall(DMSwarmGetField(swarm, "position", NULL, NULL, (void **)&pos));

  // Get an array representation of the ion velocities.
  PetscCall(DMSwarmGetField(swarm, "velocity", NULL, NULL, (void **)&vel));

  // Get the number of ions on this rank.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  // Extract cell widths.
  dx = ctx->grid.d.x;
  dy = ctx->grid.d.y;
  dz = ctx->grid.d.z;

  // Loop over ions.
  for (ip=0; ip<np; ip++) {
    // Get the current ion's coordinates.
    r = pos[ip];
    // Normalize each coordinate to a fractional number of grid cells.
    x = r.x / dx;
    y = r.y / dy;
    z = r.z / dz;
    // Compute the x-dimension neighbors and corresponding weights.
    ixl = (PetscInt)x;
    ixh = ixl+1;
    wxh = x - (PetscReal)ixl;
    wxl = 1.0 - wxh;
    // Compute the y-dimension neighbors and corresponding weights.
    iyl = (PetscInt)y;
    iyh = iyl+1;
    wyh = y - (PetscReal)iyl;
    wyl = 1.0 - wyh;
    // Compute the z-dimension neighbors and corresponding weights.
    izl = (PetscInt)z;
    izh = izl+1;
    wzh = z - (PetscReal)izl;
    wzl = 1.0 - wzh;
    // Compute the weight of each nearby grid point.
    hhh = wzh*wyh*wxh;
    lhh = wzl*wyh*wxh;
    hlh = wzh*wyl*wxh;
    llh = wzl*wyl*wxh;
    hhl = wzh*wyh*wxl;
    lhl = wzl*wyh*wxl;
    hll = wzh*wyl*wxl;
    lll = wzl*wyl*wxl;
    // Assign density values (zeroth moment).
    array[izh][iyh][ixh].n += hhh;
    array[izl][iyh][ixh].n += lhh;
    array[izh][iyl][ixh].n += hlh;
    array[izl][iyl][ixh].n += llh;
    array[izh][iyh][ixl].n += hhl;
    array[izl][iyh][ixl].n += lhl;
    array[izh][iyl][ixl].n += hll;
    array[izl][iyl][ixl].n += lll;
    // Assign flux values (first moments wrt velocity).
    v = vel[ip];
    w[0] = v.x;
    w[1] = v.y;
    w[2] = v.z;
    for (dim=0; dim<NDIM; dim++) {
      array[izh][iyh][ixh].flux[dim] += w[dim]*hhh;
      array[izl][iyh][ixh].flux[dim] += w[dim]*lhh;
      array[izh][iyl][ixh].flux[dim] += w[dim]*hlh;
      array[izl][iyl][ixh].flux[dim] += w[dim]*llh;
      array[izh][iyh][ixl].flux[dim] += w[dim]*hhl;
      array[izl][iyh][ixl].flux[dim] += w[dim]*lhl;
      array[izh][iyl][ixl].flux[dim] += w[dim]*hll;
      array[izl][iyl][ixl].flux[dim] += w[dim]*lll;
    }
  }

  // Restore the local grid array.
  PetscCall(DMDAVecRestoreArray(grid, gridvec, &array));

  // Communicate local information to the persistent global grid vector.
  PetscCall(DMLocalToGlobal(grid, gridvec, ADD_VALUES, ctx->vlasov));

  // Restore the local grid vector.
  PetscCall(DMRestoreLocalVector(grid, &gridvec));

  // Restore the ion-positions array.
  PetscCall(DMSwarmRestoreField(swarm, "position", NULL, NULL, (void **)&pos));

  // Restore the ion-velocities array.
  PetscCall(DMSwarmRestoreField(swarm, "velocity", NULL, NULL, (void **)&vel));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ComputePotential(KSP ksp, Context *ctx)
{
  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  PetscCall(KSPSolve(ksp, NULL, NULL));
  PetscCall(KSPGetSolution(ksp, &ctx->phi));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ComputeInitialPhi(KSP ksp, Vec phi, void *_ctx)
{
  // Note that this function requires this signature for use with
  // `KSPSetComputeInitialGuess`.

  PetscFunctionBeginUser;

  PetscCall(VecSet(phi, 0.0));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  Context      *user=(Context *)ctx;

  PetscFunctionBeginUser;

  PetscCall(user->rhsFunc(ksp, b, ctx));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ComputeLHS(KSP ksp, Mat J, Mat A, void *ctx)
{
  Context      *user=(Context *)ctx;

  PetscFunctionBeginUser;

  PetscCall(user->lhsFunc(ksp, J, A, ctx));

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute a vector of central differences from F.

This function was designed as the first step in computing the gradient of the
scalar function F(x, y, z) at (x0, y0, z0). It computes the numerator of each
finite-difference term using 2nd-order centered, forward, and backward
approximations. It assumes that F contains ghost nodes.
*/
static PetscErrorCode
DifferenceVector(PetscReal ***F, PetscReal x0, PetscReal y0, PetscReal z0, Grid grid, PetscReal f[NDIM])
{
  PetscInt    Nx=grid.N.x, Ny=grid.N.y, Nz=grid.N.z;
  PetscInt    ixl, ixh, iyl, iyh, izl, izh;
  PetscReal   wxh, wyh, wzh;
  PetscReal   hhh, lhh, hlh, llh, hhl, lhl, hll, lll;
  PetscReal   whh, whl, wlh, wll, Ewh, Ewl;

  PetscFunctionBeginUser;


  // NOTE: This requires either a 2-point stencil or prescribed values beyond
  // the current 1-point stencil.

  // Compute the x-dimension neighbors and corresponding weights.
  ixl = (PetscInt)x0;
  ixh = ixl+1;
  wxh = x0 - (PetscReal)ixl;
  // Compute the y-dimension neighbors and corresponding weights.
  iyl = (PetscInt)y0;
  iyh = iyl+1;
  wyh = y0 - (PetscReal)iyl;
  // Compute the z-dimension neighbors and corresponding weights.
  izl = (PetscInt)z0;
  izh = izl+1;
  wzh = z0 - (PetscReal)izl;
  // Compute the central difference in x at each grid point.
  if (ixl >= 0) {
    // 2nd-order central difference at ixl
    hhl = F[izh][iyh][ixl+1] - F[izh][iyh][ixl-1];
    lhl = F[izl][iyh][ixl+1] - F[izl][iyh][ixl-1];
    hll = F[izh][iyl][ixl+1] - F[izh][iyl][ixl-1];
    lll = F[izl][iyl][ixl+1] - F[izl][iyl][ixl-1];
  } else {
    // 2nd-order forward difference at ixl
    hhl = -1.0*F[izh][iyh][ixl+2] + 4.0*F[izh][iyh][ixl+1] - 3.0*F[izh][iyh][ixl];
    lhl = -1.0*F[izl][iyh][ixl+2] + 4.0*F[izl][iyh][ixl+1] - 3.0*F[izl][iyh][ixl];
    hll = -1.0*F[izh][iyl][ixl+2] + 4.0*F[izh][iyl][ixl+1] - 3.0*F[izh][iyl][ixl];
    lll = -1.0*F[izl][iyl][ixl+2] + 4.0*F[izl][iyl][ixl+1] - 3.0*F[izl][iyl][ixl];
  }
  if (ixh < Nx) {
    // 2nd-order central difference at ixh
    hhh = F[izh][iyh][ixh+1] - F[izh][iyh][ixh-1];
    lhh = F[izl][iyh][ixh+1] - F[izl][iyh][ixh-1];
    hlh = F[izh][iyl][ixh+1] - F[izh][iyl][ixh-1];
    llh = F[izl][iyl][ixh+1] - F[izl][iyl][ixh-1];
  } else {
    // 2nd-order backward difference at ixh
    hhh = +3.0*F[izh][iyh][ixh] - 4.0*F[izh][iyh][ixh-1] + 1.0*F[izh][iyh][ixh-2];
    lhh = +3.0*F[izl][iyh][ixh] - 4.0*F[izl][iyh][ixh-1] + 1.0*F[izl][iyh][ixh-2];
    hlh = +3.0*F[izh][iyl][ixh] - 4.0*F[izh][iyl][ixh-1] + 1.0*F[izh][iyl][ixh-2];
    llh = +3.0*F[izl][iyl][ixh] - 4.0*F[izl][iyl][ixh-1] + 1.0*F[izl][iyl][ixh-2];
  }
  whh = hlh + wyh*(hhh - hlh);
  whl = hll + wyh*(hhl - hll);
  wlh = llh + wyh*(lhh - llh);
  wll = lll + wyh*(lhl - lll);
  Ewh = wlh + wxh*(whh - wlh);
  Ewl = wll + wxh*(whl - wll);
  f[0] = Ewl + wzh*(Ewh - Ewh);
  // Compute the central difference in y at each grid point.
  if (iyl >= 0) {
    // 2nd-order central difference at iyl
    hlh = F[izh][iyl+1][ixh] - F[izh][iyl-1][ixh];
    llh = F[izl][iyl+1][ixh] - F[izl][iyl-1][ixh];
    hll = F[izh][iyl+1][ixl] - F[izh][iyl-1][ixl];
    lll = F[izl][iyl+1][ixl] - F[izl][iyl-1][ixl];
  } else {
    // 2nd-order forward difference at iyl
    hlh = -1.0*F[izh][iyl+2][ixh] + 4.0*F[izh][iyl+1][ixh] - 3.0*F[izh][iyl][ixh];
    llh = -1.0*F[izl][iyl+2][ixh] + 4.0*F[izl][iyl+1][ixh] - 3.0*F[izl][iyl][ixh];
    hll = -1.0*F[izh][iyl+2][ixl] + 4.0*F[izh][iyl+1][ixl] - 3.0*F[izh][iyl][ixl];
    lll = -1.0*F[izl][iyl+2][ixl] + 4.0*F[izl][iyl+1][ixl] - 3.0*F[izl][iyl][ixl];
  }
  if (iyh < Ny) {
    // 2nd-order central difference at iyh
    hhh = F[izh][iyh+1][ixh] - F[izh][iyh-1][ixh];
    lhh = F[izl][iyh+1][ixh] - F[izl][iyh-1][ixh];
    hhl = F[izh][iyh+1][ixl] - F[izh][iyh-1][ixl];
    lhl = F[izl][iyh+1][ixl] - F[izl][iyh-1][ixl];
  } else {
    // 2nd-order backward difference at iyh
    hhh = +3.0*F[izh][iyh][ixh] - 4.0*F[izh][iyh-1][ixh] + 1.0*F[izh][iyh-2][ixh];
    lhh = +3.0*F[izl][iyh][ixh] - 4.0*F[izl][iyh-1][ixh] + 1.0*F[izl][iyh-2][ixh];
    hhl = +3.0*F[izh][iyh][ixl] - 4.0*F[izh][iyh-1][ixl] + 1.0*F[izh][iyh-2][ixl];
    lhl = +3.0*F[izl][iyh][ixl] - 4.0*F[izl][iyh-1][ixl] + 1.0*F[izl][iyh-2][ixl];
  }
  whh = hlh + wyh*(hhh - hlh);
  whl = hll + wyh*(hhl - hll);
  wlh = llh + wyh*(lhh - llh);
  wll = lll + wyh*(lhl - lll);
  Ewh = wlh + wxh*(whh - wlh);
  Ewl = wll + wxh*(whl - wll);
  f[1] = Ewl + wzh*(Ewh - Ewh);
  // Compute the central difference in z at each grid point.
  if (izl >= 0) {
    // 2nd-order central difference at izl
    lhh = F[izl+1][iyh][ixh] - F[izl-1][iyh][ixh];
    llh = F[izl+1][iyl][ixh] - F[izl-1][iyl][ixh];
    lhl = F[izl+1][iyh][ixl] - F[izl-1][iyh][ixl];
    lll = F[izl+1][iyl][ixl] - F[izl-1][iyl][ixl];
  } else {
    // 2nd-order forward difference at izl
    lhh = -1.0*F[izl+2][iyh][ixh] + 4.0*F[izl+1][iyh][ixh] - 3.0*F[izl][iyh][ixh];
    llh = -1.0*F[izl+2][iyl][ixh] + 4.0*F[izl+1][iyl][ixh] - 3.0*F[izl][iyl][ixh];
    lhl = -1.0*F[izl+2][iyh][ixl] + 4.0*F[izl+1][iyh][ixl] - 3.0*F[izl][iyh][ixl];
    lll = -1.0*F[izl+2][iyl][ixl] + 4.0*F[izl+1][iyl][ixl] - 3.0*F[izl][iyl][ixl];
  }
  if (izh < Nz) {
    // 2nd-order central difference at izh
    hhh = F[izh+1][iyh][ixh] - F[izh-1][iyh][ixh];
    hlh = F[izh+1][iyl][ixh] - F[izh-1][iyl][ixh];
    hhl = F[izh+1][iyh][ixl] - F[izh-1][iyh][ixl];
    hll = F[izh+1][iyl][ixl] - F[izh-1][iyl][ixl];
  } else {
    // 2nd-order backward difference at izh
    hhh = +3.0*F[izh][iyh][ixh] - 4.0*F[izh-1][iyh][ixh] + 1.0*F[izh-2][iyh][ixh];
    hlh = +3.0*F[izh][iyl][ixh] - 4.0*F[izh-1][iyl][ixh] + 1.0*F[izh-2][iyl][ixh];
    hhl = +3.0*F[izh][iyh][ixl] - 4.0*F[izh-1][iyh][ixl] + 1.0*F[izh-2][iyh][ixl];
    hll = +3.0*F[izh][iyl][ixl] - 4.0*F[izh-1][iyl][ixl] + 1.0*F[izh-2][iyl][ixl];
  }
  whh = hlh + wyh*(hhh - hlh);
  whl = hll + wyh*(hhl - hll);
  wlh = llh + wyh*(lhh - llh);
  wll = lll + wyh*(lhl - lll);
  Ewh = wlh + wxh*(whh - wlh);
  Ewl = wll + wxh*(whl - wll);
  f[2] = Ewl + wzh*(Ewh - Ewh);

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute \vec{c} = \vec{a} \cdot \vec{b}. */
static PetscErrorCode
DotProduct(PetscReal a[NDIM], PetscReal b[NDIM], PetscReal *c)
{
  PetscFunctionBeginUser;
  *c = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute \vec{c} = \vec{a} \times \vec{b}. */
static PetscErrorCode
CrossProduct(PetscReal a[NDIM], PetscReal b[NDIM], PetscReal c[NDIM])
{
  PetscFunctionBeginUser;
  c[0] = a[1]*b[2] - b[2]*a[1];
  c[1] = a[2]*b[0] - b[0]*a[2];
  c[2] = a[0]*b[1] - b[1]*a[0];
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
BorisMover(KSP ksp, Context *ctx)
{
  PetscReal   q=ctx->ions.q;
  PetscReal   m=ctx->ions.m;
  PetscReal   B[NDIM]={ctx->plasma.B0.x, ctx->plasma.B0.y, ctx->plasma.B0.z};
  PetscReal   dt=ctx->dt;
  PetscInt    dim;
  PetscReal   dx=ctx->grid.d.x, dy=ctx->grid.d.y, dz=ctx->grid.d.z;
  PetscReal   h[NDIM]={1.0/dx, 1.0/dy, 1.0/dz};
  PetscReal   t[NDIM], s[NDIM], t_dot_t;
  PetscReal   tscale, Escale[NDIM];
  DM          swarm=ctx->swarm;
  DM          phiDM;
  Vec         phiGlobal, phiLocal;
  PetscReal   ***phi;
  PetscInt    np, ip;
  RealVector  r, *pos, v, *vel;
  PetscReal   x, y, z;
  PetscReal   E[NDIM]={0.0, 0.0, 0.0};
  PetscReal   vold[NDIM];
  PetscReal   vminus[NDIM], vprime[NDIM], vplus[NDIM];
  PetscReal   vminus_cross_t[NDIM], vprime_cross_s[NDIM];

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  /* Compute \vec{t} = \frac{q\vec{B}}{m}\frac{\Delta t}{2}. */
  tscale = 0.5 * (q/m) * dt;
  for (dim=0; dim<NDIM; dim++) {
    t[dim] = tscale * B[dim];
  }

  /* Compute \vec{s} = \frac{2\vec{t}}{1 + \vec{t}\cdot\vec{t}}. */
  PetscCall(DotProduct(t, t, &t_dot_t));
  for (dim=0; dim<NDIM; dim++) {
    s[dim] = 2.0 * t[dim] / (1 + t_dot_t);
  }

  /* Compute the electric-field scale factors.

  These account for the species constants as well as the 2nd-order
  finite-difference gradient scale factors.
  */
  for (dim=0; dim<NDIM; dim++) {
    Escale[dim] = -0.5 * h[dim] * tscale;
  }

  /* Get a local copy of phi with ghost cells. */
  PetscCall(KSPGetSolution(ksp, &phiGlobal));
  PetscCall(KSPGetDM(ksp, &phiDM));
  PetscCall(DMGetLocalVector(phiDM, &phiLocal));
  PetscCall(DMGlobalToLocal(phiDM, phiGlobal, INSERT_VALUES, phiLocal));

  /* Get a temporary array representing the local electrostatic potential. */
  PetscCall(DMDAVecGetArray(phiDM, phiLocal, &phi));

  /* Get the number of local particles. */
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  /* Get an array representation of the particle positions. */
  PetscCall(DMSwarmGetField(swarm, "position", NULL, NULL, (void **)&pos));

  /* Get an array representation of the particle velocities. */
  PetscCall(DMSwarmGetField(swarm, "velocity", NULL, NULL, (void **)&vel));

  /* Loop over particles and interpolate E to grid points. */
  for (ip=0; ip<np; ip++) {
    /* Get the current particle's coordinates. */
    r = pos[ip];
    v = vel[ip];
    /* Normalize each coordinate to a fractional number of grid cells. */
    x = r.x / dx;
    y = r.y / dy;
    z = r.z / dz;
    /* Compute the electric field due to this particle: \vec{E} = -\nabla\phi. */
    PetscCall(DifferenceVector(phi, x, y, z, ctx->grid, E));
    /* Compute \vec{v}^-. */
    vold[0] = v.x;
    vold[1] = v.y;
    vold[2] = v.z;
    for (dim=0; dim<NDIM; dim++) {
      vminus[0] = vold[dim] + Escale[dim]*E[dim];
    }
    /* Compute \vec{v}^- \times \vec{t}. */
    PetscCall(CrossProduct(vminus, t, vminus_cross_t));
    /* Compute \vec{v}^\prime = \vec{v}^- + \vec{v}^- \times \vec{t}. */
    for (dim=0; dim<NDIM; dim++) {
      vprime[dim] = vminus[dim] + vminus_cross_t[dim];
    }
    /* Compute \vec{v}^\prime \times \vec{s}. */
    PetscCall(CrossProduct(vprime, s, vprime_cross_s));
    /* Compute \vec{v}^+ = \vec{v}^- + \vec{v}^\prime \times \vec{s}. */
    for (dim=0; dim<NDIM; dim++) {
      vplus[dim] = vminus[dim] + vprime_cross_s[dim];
    }
    /* Assign new particle velocities. */
    v.x = vplus[0] + Escale[0]*E[0];
    v.y = vplus[1] + Escale[1]*E[1];
    v.z = vplus[2] + Escale[2]*E[2];
    vel[ip] = v;
  }

  // Restore the particle-positions array.
  PetscCall(DMSwarmRestoreField(swarm, "position", NULL, NULL, (void **)&pos));

  // Restore the particle-velocities array.
  PetscCall(DMSwarmRestoreField(swarm, "velocity", NULL, NULL, (void **)&vel));

  // Restore the borrowed potential array.
  PetscCall(DMDAVecRestoreArray(phiDM, phiLocal, &phi));

  // Restore the borrowed local phi vector.
  PetscCall(DMRestoreLocalVector(phiDM, &phiLocal));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Basic elastic-scattering routine for ion-neutral collisions.

This function is based on, but not identical to, EPPIC elastic_scatter.
*/
static PetscErrorCode
CollideParticles(Context *ctx)
{
  PetscInt   Nc;                                      // the number of collisions to attempt
  PetscInt   Ns=0;                                    // the number of successful collision attempts
  PetscInt   Nf=0;                                    // the number of failed collision attempts
  PetscInt   Np=ctx->plasma.Np;                       // the total number of ions
  PetscInt   ip;                                      // the current ion
  PetscReal  fc=ctx->ions.nu * ctx->dt;               // the product of the collision rate and the time step
  PetscReal  viT=ctx->ions.vT.r;                      // the ion-species thermal speed
  PetscReal  vi0x=ctx->ions.v0.x;                     // the ion-species x-axis drift component
  PetscReal  vi0y=ctx->ions.v0.y;                     // the ion-species y-axis drift component
  PetscReal  vi0z=ctx->ions.v0.z;                     // the ion-species z-axis drift component
  PetscReal  vnT=ctx->neutrals.vT;                    // the neutral-species thermal speed
  PetscReal  vn0=ctx->neutrals.v0.r;                  // the neutral-species drift speed
  PetscReal  vn0x=ctx->neutrals.v0.x;                 // the neutral-species x-axis drift component
  PetscReal  vn0y=ctx->neutrals.v0.y;                 // the neutral-species y-axis drift component
  PetscReal  vn0z=ctx->neutrals.v0.z;                 // the neutral-species z-axis drift component
  PetscReal  vrm;                                     // the maximum ion-neutral relative velocity
  PetscReal  mi=ctx->ions.m;                          // the ion-species mass
  PetscReal  mn=ctx->neutrals.m;                      // the neutral-species mass
  PetscReal  M=mn+mi;                                 // the total mass (mi+mn)
  DM         swarm=ctx->swarm;
  RealVector *vel;
  PetscReal  vnx, vny, vnz;                           // neutral-particle velocity components
  PetscReal  vix, viy, viz;                           // ion velocity components
  PetscReal  vrx, vry, vrz;                           // ion-neutral relative-velocity components
  PetscReal  vrr;                                     // ion-neutral relative-velocity magnitude
  PetscReal  vcx, vcy, vcz;                           // center-of-mass velocity components
  PetscReal  vcr;                                     // the ion speed with respect to the center of mass
  PetscReal  costht, sintht, cosphi, sinphi;
  PetscReal  ux, uy, uz, uperp, uphi;                 // components of the unit scattering vector
  PetscReal  ux1, uy1, uz1, ux2, uy2, uz2;
  PetscReal  vfx, vfy, vfz, vfr;                      // components and magnitude of current ion's final speed
  long       seed=ctx->seed;
  PetscReal  ratio;                                   // ratio of current ion's final speed to thermal speed

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Compute the maximum relative velocity.
  vrm = 4.0*viT + PetscSqrtReal(PetscSqr(vi0x-vn0x) + PetscSqr(vi0y-vn0y) + PetscSqr(vi0z-vn0z));

  // Compute the number of collisions to attempt.
  Nc = (PetscInt)(((PetscReal)Np * fc * vrm) / ((vnT + vn0) * PetscSqrtReal(mn / mi)));
  if (Nc > Np) {
    Nc = Np;
  } else if (Nc < 0) {
    Nc = 0;
  }

  // Get an array representation of the particle velocities.
  PetscCall(DMSwarmGetField(swarm, "velocity", NULL, NULL, (void **)&vel));

  // Report number of collisions.
  PRINT_WORLD("Colliding %d particles out of %d ...\n", Nc, Np);

  // Attempt collisions until we reach the required number.
  while (Ns < Nc) {
    // Choose a random ion from the full distribution.
    ip = (PetscInt)(Np*ran3(&seed));

    // Store the ion velocity components.
    vix = vel[ip].x;
    viy = vel[ip].y;
    viz = vel[ip].z;

    // Choose a random neutral-particle velocity.
    vnx = gasdev(&seed)*vnT + vn0x;
    vny = gasdev(&seed)*vnT + vn0y;
    vnz = gasdev(&seed)*vnT + vn0z;

    // Compute the ion-neutral relative velocity.
    vrx = vix - vnx;
    vry = viy - vny;
    vrz = viz - vnz;
    vrr = PetscSqrtReal(PetscSqr(vrx) + PetscSqr(vry) + PetscSqr(vrz));

    // Collide only if the squared relative velocity is greater than the square
    // of a random percentage of the maximum relative velocity.
    if (PetscSqr(ran3(&seed)) < PetscSqr(vrr) / PetscSqr(vrm)) {

      // Compute the center-of-mass (CoM) velocity.
      vcx = (vix*mi + vnx*mn) / M;
      vcy = (viy*mi + vny*mn) / M;
      vcz = (viz*mi + vnz*mn) / M;

      // Compute the particle speed relative to the CoM velocity.
      vcr = PetscSqrtReal(PetscSqr(vix-vcx) + PetscSqr(viy-vcy) + PetscSqr(viz-vcz));

      // Compute the unit scattering vector relative to the CoM z axis.
      uz = 2.0*ran3(&seed) - 1.0;
      uperp = PetscSqrtReal(1.0 - PetscSqr(uz));
      uphi = 2*PETSC_PI * ran3(&seed);
      ux = uperp*PetscCosReal(uphi);
      uy = uperp*PetscSinReal(uphi);

      // Rotate the CoM frame to align its z axis with the incident direction.
      costht = vrz / vrr;
      sintht = PetscSqrtReal(1.0 - PetscSqr(costht));
      cosphi = vrx / (vrr*sintht);
      sinphi = vry / (vrr*sintht);

      /* Rotate the unit scattering vector to the incident coordinate system.
      1. rotation about CoM y axis:          (xc, yc, zc) -> (xp, yp, zp)
      2. rotation about intermediate z axis: (xp, yp, zp) -> (xi, yi, zi)
      */
      ux1 = uz*sintht + ux*costht;
      uy1 = uy;
      uz1 = uz*costht - ux*sintht;
      ux2 = ux1*cosphi - uy1*sinphi;
      uy2 = ux1*sinphi + uy1*cosphi;
      uz2 = uz1;

      /* Assign final CoM velocity components.
      vfx = vcr * ((uz*sintht + ux*costht)*cosphi - uy*sinphi)
      vfy = vcr * ((uz*sintht - ux*costht)*sinphi + uy*cosphi)
      vfz = vcr * ( uz*costht - ux*sintht                    )
      */
      vfx = vcx + vcr*ux2;
      vfy = vcy + vcr*uy2;
      vfz = vcz + vcr*uz2;

      /* Finalize
      - if result is unphysical, do not use it
      - otherwise, update the ion velocity components and count the collision
      */
      vfr = PetscSqrtReal(PetscSqr(vfx) + PetscSqr(vfy) + PetscSqr(vfz));
      ratio = vfr / viT;
      if (ratio > 10) {
        PRINT_WORLD("Warning: Refusing to accept collision that results in final speed = %4.1f times thermal speed\n", ratio);
        Nf++;
        // TODO: We may want to terminate after a certain number of failures, or
        // keep more detailed statistics of the types of failures.
      } else {
        vel[ip].x = vfx;
        vel[ip].y = vfy;
        vel[ip].z = vfz;
        Ns++;
      }

    }
  }

  // Restore the particle-velocities array.
  PetscCall(DMSwarmRestoreField(swarm, "velocity", NULL, NULL, (void **)&vel));

  // Report the number of actual collisions.
  PRINT_WORLD("Collision efficiency: %6.4f.\n", (PetscReal)Ns/(PetscReal)(Ns+Nf));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
UpdateVelocities(KSP ksp, Context *ctx)
{
  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Apply the Boris mover to integrate dv/dt = E + vxB.
  PetscCall(BorisMover(ksp, ctx));

  // Apply the appropriate collision algorithm.
  if (ctx->neutrals.m > 0.0) {
    PetscCall(CollideParticles(ctx));
  }

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Update the ion positions according to $\frac{d\vec{r}}{dt} = \vec{v}$. */
static PetscErrorCode
UpdatePositions(Context *ctx)
{
  DM          swarm=ctx->swarm;
  RealVector  *pos, *vel;
  PetscInt    ip, np;
  PetscReal   x, y, z;
  PetscReal   Lx=ctx->grid.L.x;
  PetscReal   Ly=ctx->grid.L.y;
  PetscReal   Lz=ctx->grid.L.z;
  PetscReal   x0=ctx->grid.p0.x;
  PetscReal   y0=ctx->grid.p0.y;
  PetscReal   z0=ctx->grid.p0.z;
  PetscReal   x1=ctx->grid.p1.x;
  PetscReal   y1=ctx->grid.p1.y;
  PetscReal   z1=ctx->grid.p1.z;
  PetscReal   dt=ctx->dt;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get an array representation of the ion positions.
  PetscCall(DMSwarmGetField(swarm, "position", NULL, NULL, (void **)&pos));

  // Get an array representation of the ion velocities.
  PetscCall(DMSwarmGetField(swarm, "velocity", NULL, NULL, (void **)&vel));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  // Loop over ions.
  for (ip=0; ip<np; ip++) {
    // Update the x position.
    x = pos[ip].x + vel[ip].x*dt;
    // [DEV] Apply periodic BC along x.
    while (x < x0) {
      x += Lx;
    }
    while (x > x1) {
      x -= Lx;
    }
    // Update the y position.
    y = pos[ip].y + vel[ip].y*dt;
    // [DEV] Apply periodic BC along y.
    while (y < y0) {
      y += Ly;
    }
    while (y > y1) {
      y -= Ly;
    }
    // Update the z position.
    z = pos[ip].z + vel[ip].z*dt;
    // [DEV] Apply periodic BC along z.
    while (z < z0) {
      z += Lz;
    }
    while (z > z1) {
      z -= Lz;
    }
    // TODO: Implement more flexible BC.
  }
  // Restore the ion-positions array.
  PetscCall(DMSwarmRestoreField(swarm, "position", NULL, NULL, (void **)&pos));

  // Restore the ion-velocities array.
  PetscCall(DMSwarmRestoreField(swarm, "velocity", NULL, NULL, (void **)&vel));

  // Update the swarm.
  PetscCall(DMSwarmMigrate(swarm, PETSC_TRUE));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  MPIContext  mpi;
  time_t      startTime, endTime;
  Context     ctx;
  DM          grid, solve;
  KSP         ksp;
  PetscInt    it;
  char        itfmt[5];
  char        pathfmt[PETSC_MAX_PATH_LEN]="", pathstr[PETSC_MAX_PATH_LEN];
  char        stepfmt[256]="", stepstr[256];
  PetscInt    itwidth;

  PetscFunctionBeginUser;

  /* Initialize PETSc and MPI. */
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &mpi.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi.size));
  if (mpi.rank == 0) {
    time(&startTime);
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n**************** START *****************\n\n"));

  /* Assign parameter values from user arguments or defaults. */
  PetscCall(ProcessOptions(&ctx));

  /* Store MPI information in the application context. */
  ctx.mpi = mpi;

  /* Define a rank-specific random-number seed. */
  ctx.seed = (long)(-(mpi.rank + 1)*12345);

  /* Set up discrete grid. */
  PetscCall(InitializeGridDM(&grid, &ctx));
  PetscCall(DMCreateGlobalVector(grid, &ctx.vlasov));
  PetscCall(VecZeroEntries(ctx.vlasov));

  /* Set up particle swarm. */
  PetscCall(InitializeSwarmDM(grid, &ctx));

  /* Set initial particle positions. */
  PetscCall(InitializePositions(&ctx));

  /* Set initial particle velocities. */
  PetscCall(InitializeVelocities(&ctx));

  /* Echo the initial state. */
  if (mpi.rank == 0) {
    PetscCall(EchoSetup(ctx));
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  /* Compute initial density and flux. */
  PetscCall(CollectParticles(&ctx));

  /* Compute initial electrostatic potential. */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(InitializePotentialDM(grid, &solve));
  PetscCall(KSPSetDM(ksp, solve));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetComputeInitialGuess(ksp, ComputeInitialPhi, &ctx));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, &ctx));
  PetscCall(KSPSetComputeOperators(ksp, ComputeLHS, &ctx));
  PetscCall(ComputePotential(ksp, &ctx));

  /* Create a format string for the time step. */
  itwidth = 1+PetscLog10Real(ctx.Nt);
  sprintf(itfmt, "%%0%dd", itwidth);

  /* Create a template for time-dependent filenames. */
  PetscCall(PetscStrcat(pathfmt, "arrays-"));
  PetscCall(PetscStrcat(pathfmt, itfmt));
  PetscCall(PetscStrcat(pathfmt, ".hdf"));

  /* Output initial conditions. */
  sprintf(pathstr, "arrays-initial.hdf");
  PetscCall(OutputHDF5(pathstr, &ctx));

  /* Create a template for the time-step string. */
  PetscCall(PetscStrcat(stepfmt, "Time step "));
  PetscCall(PetscStrcat(stepfmt, itfmt));
  PetscCall(PetscStrcat(stepfmt, "\n"));

  PRINT_WORLD("\n*** Main time-step loop ***\n\n");
  /* Begin main time-step loop. */
  for (it=0; it<ctx.Nt; it++) {

    /* Create a string to display time step with the appropriate width. */
    sprintf(stepstr, stepfmt, it);
    PRINT_WORLD(stepstr);

    /* Update velocities */
    PetscCall(UpdateVelocities(ksp, &ctx));

    /* Update positions. */
    PetscCall(UpdatePositions(&ctx));

    /* Compute density and flux from ion positions. */
    PetscCall(CollectParticles(&ctx));

    /* Compute potential from density. */
    PetscCall(ComputePotential(ksp, &ctx));

    /* Output current time step. */
    sprintf(pathstr, pathfmt, it);
    PetscCall(OutputHDF5(pathstr, &ctx));

  }

  /* Free memory. */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&ctx.vlasov));
  PetscCall(DMDestroy(&grid));
  PetscCall(DMDestroy(&ctx.swarm));
  PetscCall(DMDestroy(&solve));

  /* Write time information. */
  if (mpi.rank == 0) {
    time(&endTime);
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n----------------------------------------\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Start time: %s", asctime(localtime(&startTime))));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "End time:   %s", asctime(localtime(&endTime))));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "----------------------------------------\n"));

  /* Finalize PETSc and MPI. */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n***************** END ******************\n"));
  PetscCall(PetscFinalize());

  return 0;
}