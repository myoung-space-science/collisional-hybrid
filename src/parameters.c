#include <petsc.h>
#include "hybrid.h"
#include "lhs.h"
#include "rhs.h"

/* Supported RHS functions. */
const char *RHSTypes[] = {
  "constant", "sinusoidal", "full", "RHSType", "RHS_", NULL
};

/* Supported LHS functions. */
const char *LHSTypes[] = {
  "identity", "laplacian", "full", "LHSType", "LHS_", NULL
};

/* Supported density functions. */
const char *DensityTypes[] ={
  "flat-sobol", "flat-reverse", "flat-normal", "uniform", "sinusoidal", "gaussian", "DensityType", "DENSITY_", NULL
};

/* Supported boundary conditions. */
const char *BCTypes[] = {
  "periodic", "injection", "reflection", "dirichlet", "neumann", "BCType", "BC_", NULL
};

/* Read runtime arguments from the command line or a file. */
PetscErrorCode ProcessOptions(Context *ctx)
{
  PetscInt  intArg;
  PetscReal realArg;
  PetscBool boolArg;
  PetscEnum enumArg;
  PetscBool found;
  PetscReal tmp;

  PetscFunctionBeginUser;

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
  // Set up boundary conditions.
  // - If one is periodic, both must be periodic.
  // - The pair of boundary conditions for a given axis sets the DM boundary
  //   type for that axis.
  // - The combination of all boundary conditions determines the LHS stencil
  //   function.
  // - Some combinations may not be implemented.
  if ((ctx->xBC[0] == BC_PERIODIC) && (ctx->xBC[1] == BC_PERIODIC)) {
    ctx->xDMBC = DM_BOUNDARY_PERIODIC;
  } else if ((ctx->xBC[0] != BC_PERIODIC) && (ctx->xBC[1] != BC_PERIODIC)) {
    ctx->xDMBC = DM_BOUNDARY_GHOSTED;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Inconsistent x-axis boundary conditions: {%s, %s}\n", BCTypes[ctx->xBC[0]], BCTypes[ctx->xBC[1]]);
  }
  if ((ctx->yBC[0] == BC_PERIODIC) && (ctx->yBC[1] == BC_PERIODIC)) {
    ctx->yDMBC = DM_BOUNDARY_PERIODIC;
  } else if ((ctx->yBC[0] != BC_PERIODIC) && (ctx->yBC[1] != BC_PERIODIC)) {
    ctx->yDMBC = DM_BOUNDARY_GHOSTED;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Inconsistent y-axis boundary conditions: {%s, %s}\n", BCTypes[ctx->yBC[0]], BCTypes[ctx->yBC[1]]);
  }
  if ((ctx->zBC[0] == BC_PERIODIC) && (ctx->zBC[1] == BC_PERIODIC)) {
    ctx->zDMBC = DM_BOUNDARY_PERIODIC;
  } else if ((ctx->zBC[0] != BC_PERIODIC) && (ctx->zBC[1] != BC_PERIODIC)) {
    ctx->zDMBC = DM_BOUNDARY_GHOSTED;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Inconsistent z-axis boundary conditions: {%s, %s}\n", BCTypes[ctx->zBC[0]], BCTypes[ctx->zBC[1]]);
  }
  if ((ctx->xDMBC == DM_BOUNDARY_PERIODIC) && (ctx->yDMBC == DM_BOUNDARY_PERIODIC) && (ctx->zDMBC == DM_BOUNDARY_PERIODIC)) {
    ctx->stencilFunc = ComputePeriodicStencil;
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Unsupported boundary conditions: {%s, %s} x {%s, %s} x {%s, %s}\n", BCTypes[ctx->xBC[0]], BCTypes[ctx->xBC[1]], BCTypes[ctx->yBC[0]], BCTypes[ctx->yBC[1]], BCTypes[ctx->zBC[0]], BCTypes[ctx->zBC[1]]);
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


/* Report the current value of each runtime parameter. */
PetscErrorCode EchoSetup(Context ctx)
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
  PetscCall(PetscViewerASCIIPrintf(viewer, "xBC = {%s, %s}\n", BCTypes[ctx.xBC[0]], BCTypes[ctx.xBC[1]]));
  PetscCall(PetscViewerASCIIPrintf(viewer, "yBC = {%s, %s}\n", BCTypes[ctx.yBC[0]], BCTypes[ctx.yBC[1]]));
  PetscCall(PetscViewerASCIIPrintf(viewer, "zBC = {%s, %s}\n", BCTypes[ctx.zBC[0]], BCTypes[ctx.zBC[1]]));
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


