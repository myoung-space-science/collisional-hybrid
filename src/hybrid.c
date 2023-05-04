/* 3D Hybrid PIC */
static char help[] = "A 3D hybrid particle-in-cell (PIC) simulation.";

#include <time.h>
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include <petscviewerhdf5.h>

#define PUSH_FUNC {PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n--> Entering %s(...) <--\n\n", __func__));}
#define POP_FUNC {PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n--> Exiting %s(...) <--\n\n", __func__));}

#define NDIM 3
#define Q 1.6022e-19   // fundamental charge in C
#define MP 1.6726e-27  // proton mass in kg
#define ME 9.1094e-31  // electron mass in kg
#define KB  1.3807e-23 // Boltzmann constant

// Temporary declaration of number of particles per cell, per dimension.
#define NPPCELL 1
// Temporary declaration of total number of particles per distribution.
#define NPTOTAL 32768

// The number of non-null values in the LHS-operator matrix stencil. This is
// equivalent to the standard box stencil with the corners removed.
#define NVALUES 19

const char *RHSTypes[] = {
  "constant", "sinusoidal", "full", "RHSType", "RHS_", NULL
};
typedef enum {
  RHS_CONSTANT,
  RHS_SINUSOIDAL,
  RHS_FULL,
} RHSType;

const char *LHSTypes[] = {
  "identity", "laplacian", "full", "LHSType", "LHS_", NULL
};
typedef enum {
  LHS_IDENTITY,
  LHS_LAPLACIAN,
  LHS_FULL,
} LHSType;

typedef struct {
  PetscInt x; // x component
  PetscInt y; // y component
  PetscInt z; // z component
} IntVector;

typedef struct {
  PetscReal x; // x component
  PetscReal y; // y component
  PetscReal z; // z component
} RealVector;

typedef struct {
  IntVector  N;  // number of cells
  RealVector L;  // physical length
  RealVector d;  // physical cell spacing
  RealVector p0; // lower physical bound
  RealVector p1; // upper physical bound
} Grid;

typedef struct {
  PetscReal  q;     // charge
  PetscReal  m;     // mass
  PetscReal  nu;    // frequency of collisions with neutral particles
  RealVector Omega; // gyrofrequency components
  RealVector kappa; // magnetization components
  RealVector v0;    // drift velocity
  RealVector vT;    // thermal velocity
  PetscReal  gamma; // thermal coefficient
  PetscReal  T;     // temperature
} Species;

typedef struct {
  RealVector B0; // constant magnetic-field amplitude
  RealVector E0; // constant electric-field amplitude
  PetscInt   Np; // number of charged particles
} Plasma;

typedef struct {
  PetscMPIInt rank; // global processor number
  PetscMPIInt size; // total number of processors
} MPIContext;

typedef struct {
  Grid       grid;         // grid information
  Species    electrons;    // electron parameter values
  Species    ions;         // ion parameter values
  Plasma     plasma;       // plasma information
  MPIContext mpi;          // MPI information
  Vec        global;       // full vector of all Vlasov quantities
  DM         swarm;        // PIC-swarm data manager
  PetscViewer gridView;    // viewer for arrays of simulated quantities
  PetscViewer optionsView; // viewer for parameter values
  PetscBool viewLHS;       // option to view LHS operator structure
  RHSType   rhsType;       // type of RHS vector to use
  LHSType   lhsType;       // type of LHS operator to use
} Context;

typedef struct {
  PetscScalar n;          // scalar density
  PetscScalar flux[NDIM]; // vector flux
} GridNode;

static PetscErrorCode
ProcessOptions(Context *ctx)
{
  /* This could read all options from the command line, but we may want to use
  libconfig to read most of the options from a user-provided file, and only
  support a few runtime CLI options (e.g., output directory). Note that the
  third argument to PetscInitialize is the name of an options-database file, so
  it may be possible to specify all runtime options that way. */

  PetscFunctionBeginUser;

  PetscInt intArg;
  PetscReal realArg;
  PetscBool boolArg;
  PetscEnum enumArg;
  PetscBool found;

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
  PetscCall(PetscOptionsGetEnum(
            NULL, NULL, "--rhs-type", RHSTypes, &enumArg, &found));
  if (found) {
    ctx->rhsType = enumArg;
  } else {
    ctx->rhsType = RHS_FULL;
  }
  PetscCall(PetscOptionsGetEnum(
            NULL, NULL, "--lhs-type", LHSTypes, &enumArg, &found));
  if (found) {
    ctx->lhsType = enumArg;
  } else {
    ctx->lhsType = LHS_FULL;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Nx", &intArg, &found));
  if (found) {
    if (intArg < 0) {
      PetscCall(PetscPrintf(
                PETSC_COMM_WORLD,
                "Warning: Ignoring negative value for Nx: %d\n", intArg));
    } else {
      ctx->grid.N.x = intArg;
    }
  } else {
    ctx->grid.N.x = -1;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Ny", &intArg, &found));
  if (found) {
    if (intArg < 0) {
      PetscCall(PetscPrintf(
                PETSC_COMM_WORLD,
                "Warning: Ignoring negative value for Ny: %d\n", intArg));
    } else {
      ctx->grid.N.y = intArg;
    }
  } else {
    ctx->grid.N.y = -1;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Nz", &intArg, &found));
  if (found) {
    if (intArg < 0) {
      PetscCall(PetscPrintf(
                PETSC_COMM_WORLD,
                "Warning: Ignoring negative value for Nz: %d\n", intArg));
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
      PetscCall(PetscPrintf(
                PETSC_COMM_WORLD,
                "Warning: zero-width x dimension\n"));
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-y1", &realArg, &found));
  if (found) {
    ctx->grid.p1.y = realArg;
  } else {
    ctx->grid.p1.y = 1.0;
  }
  if (ctx->grid.p1.y == ctx->grid.p0.y) {
      PetscCall(PetscPrintf(
                PETSC_COMM_WORLD,
                "Warning: zero-width y dimension\n"));
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-z1", &realArg, &found));
  if (found) {
    ctx->grid.p1.z = realArg;
  } else {
    ctx->grid.p1.z = 1.0;
  }
  if (ctx->grid.p1.z == ctx->grid.p0.z) {
      PetscCall(PetscPrintf(
                PETSC_COMM_WORLD,
                "Warning: zero-width z dimension\n"));
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-Np", &intArg, &found));
  if (found) {
    ctx->plasma.Np = intArg;
  } else {
    ctx->plasma.Np = -1;
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
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ve0x", &realArg, &found));
  if (found) {
    ctx->electrons.v0.x = realArg;
  } else {
    ctx->electrons.v0.x = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ve0y", &realArg, &found));
  if (found) {
    ctx->electrons.v0.y = realArg;
  } else {
    ctx->electrons.v0.y = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-ve0z", &realArg, &found));
  if (found) {
    ctx->electrons.v0.z = realArg;
  } else {
    ctx->electrons.v0.z = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vi0x", &realArg, &found));
  if (found) {
    ctx->ions.v0.x = realArg;
  } else {
    ctx->ions.v0.x = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vi0y", &realArg, &found));
  if (found) {
    ctx->ions.v0.y = realArg;
  } else {
    ctx->ions.v0.y = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vi0z", &realArg, &found));
  if (found) {
    ctx->ions.v0.z = realArg;
  } else {
    ctx->ions.v0.z = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-veTx", &realArg, &found));
  if (found) {
    ctx->electrons.vT.x = realArg;
  } else {
    ctx->electrons.vT.x = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-veTy", &realArg, &found));
  if (found) {
    ctx->electrons.vT.y = realArg;
  } else {
    ctx->electrons.vT.y = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-veTz", &realArg, &found));
  if (found) {
    ctx->electrons.vT.z = realArg;
  } else {
    ctx->electrons.vT.z = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-viTx", &realArg, &found));
  if (found) {
    ctx->ions.vT.x = realArg;
  } else {
    ctx->ions.vT.x = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-viTy", &realArg, &found));
  if (found) {
    ctx->ions.vT.y = realArg;
  } else {
    ctx->ions.vT.y = 0.0;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-viTz", &realArg, &found));
  if (found) {
    ctx->ions.vT.z = realArg;
  } else {
    ctx->ions.vT.z = 0.0;
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

  // Set grid lengths from lower and upper bounds.
  ctx->grid.L.x = ctx->grid.p1.x - ctx->grid.p0.x;
  ctx->grid.L.y = ctx->grid.p1.y - ctx->grid.p0.y;
  ctx->grid.L.z = ctx->grid.p1.z - ctx->grid.p0.z;
  // Set species gyrofrequency from q, B0, and m.
  ctx->electrons.Omega.x = 
    PetscAbsReal(ctx->electrons.q * ctx->plasma.B0.x / ctx->electrons.m);
  ctx->electrons.Omega.y = 
    PetscAbsReal(ctx->electrons.q * ctx->plasma.B0.y / ctx->electrons.m);
  ctx->electrons.Omega.z = 
    PetscAbsReal(ctx->electrons.q * ctx->plasma.B0.z / ctx->electrons.m);
  ctx->ions.Omega.x = 
    PetscAbsReal(ctx->ions.q * ctx->plasma.B0.x / ctx->ions.m);
  ctx->ions.Omega.y = 
    PetscAbsReal(ctx->ions.q * ctx->plasma.B0.y / ctx->ions.m);
  ctx->ions.Omega.z = 
    PetscAbsReal(ctx->ions.q * ctx->plasma.B0.z / ctx->ions.m);
  // Set species magnetization from Omega and nu.
  ctx->electrons.kappa.x = ctx->electrons.Omega.x / ctx->electrons.nu;
  ctx->electrons.kappa.y = ctx->electrons.Omega.y / ctx->electrons.nu;
  ctx->electrons.kappa.z = ctx->electrons.Omega.z / ctx->electrons.nu;
  ctx->ions.kappa.x = ctx->ions.Omega.x / ctx->ions.nu;
  ctx->ions.kappa.y = ctx->ions.Omega.y / ctx->ions.nu;
  ctx->ions.kappa.z = ctx->ions.Omega.z / ctx->ions.nu;
  // Set species temperature from fluid velocities.
  ctx->electrons.T = (0.5 * ctx->electrons.m / KB)
    * ( PetscSqr(ctx->electrons.vT.x)
      + PetscSqr(ctx->electrons.vT.y)
      + PetscSqr(ctx->electrons.vT.z));
  ctx->ions.T = (0.5 * ctx->ions.m / KB)
    * ( PetscSqr(ctx->ions.vT.x)
      + PetscSqr(ctx->ions.vT.y)
      + PetscSqr(ctx->ions.vT.z));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeGridDM(DM *grid, Context *ctx)
{
  PetscInt       Nx=(ctx->grid.N.x > 0 ? ctx->grid.N.x : 7);
  PetscInt       Ny=(ctx->grid.N.y > 0 ? ctx->grid.N.y : 7);
  PetscInt       Nz=(ctx->grid.N.z > 0 ? ctx->grid.N.z : 7);
  DMBoundaryType xBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType yBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType zBC=DM_BOUNDARY_PERIODIC;
  PetscInt       dof=4;
  PetscInt       width=1;

  PetscFunctionBeginUser;

  // Create the grid DM.
  PetscCall(DMDACreate3d(
            PETSC_COMM_WORLD,
            xBC, yBC, zBC,
            DMDA_STENCIL_BOX,
            Nx, Ny, Nz,
            PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
            dof, width,
            NULL, NULL, NULL,
            grid));
  // Perform basic setup.
  PetscCall(DMDASetElementType(*grid, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(*grid));
  PetscCall(DMSetUp(*grid));
  // Synchronize values of Nx, Ny, and Nz passed via -n{x,y,z} or
  // -da_grid_{x,y,z}. Note that this gives precedence to the latter.
  PetscCall(DMDAGetInfo(
            *grid, NULL,
            &Nx, &Ny, &Nz,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  // Update the grid context where necessary.
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
  ctx->grid.d.x = 1.0 / (PetscReal)ctx->grid.N.x;
  ctx->grid.d.y = 1.0 / (PetscReal)ctx->grid.N.y;
  ctx->grid.d.z = 1.0 / (PetscReal)ctx->grid.N.z;
  // Set uniform coordinates on the grid DM.
  PetscCall(DMDASetUniformCoordinates(
            *grid,
            ctx->grid.p0.x, ctx->grid.p1.x,
            ctx->grid.p0.y, ctx->grid.p1.y,
            ctx->grid.p0.z, ctx->grid.p1.z));
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
  PetscCall(DMSwarmRegisterUserStructField(
            swarm, "position", sizeof(RealVector)));
  // --> (x, y, z) velocity components
  PetscCall(DMSwarmRegisterUserStructField(
            swarm, "velocity", sizeof(RealVector)));
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


// Type to be used for cumulative distribution functions.
typedef PetscErrorCode
(*CDF)(PetscReal x, PetscReal y, PetscReal z, PetscReal *v, Context *ctx);


static CDF
SinusoidalDensity(PetscReal x, PetscReal y, PetscReal z,
                  PetscReal *v, Context *ctx)
{
  PetscReal fx, fy, fz;

  PetscFunctionBeginUser;

  fx = PetscSinReal(2*PETSC_PI * x);
  fy = PetscSinReal(2*PETSC_PI * y);
  fz = PetscSinReal(2*PETSC_PI * z);
  *v = 1.0 + 0.25*fx*fy*fz;

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
Rejection(CDF density, Context *ctx)
{
  PetscRandom random;
  PetscRandom rx, ry, rz, rv;
  PetscReal   xr, yr, zr, vr;
  PetscInt    np, ip, Np=0;
  DM          swarm=ctx->swarm;
  PetscScalar *coords;
  PetscReal   X, Y, Z;
  PetscReal   x, y, z;
  PetscReal   v, V;

  // [DEV]
  // - This needs to look like InitializeSwarmCoordinates.
  // - It cannot assume the ctx->plasma.Np is correct.
  // - It cannot access the swarm 'position' field.
  PetscFunctionBeginUser;

  // Get a representation of the particle coordinates.
  PetscCall(DMSwarmGetField(
            swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rx));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &ry));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rz));
  PetscCall(PetscRandomSetSeed(rx, (unsigned long)ctx->mpi.rank));
  PetscCall(PetscRandomSetSeed(ry, (unsigned long)ctx->mpi.rank));
  PetscCall(PetscRandomSetSeed(rz, (unsigned long)ctx->mpi.rank));
  // A while loop may be more appropriate. See bottom of EPPIC rejectND.cc
  for (ip=0; ip<np; ip++) {
    x = 0.0;
    y = 0.0;
    z = 0.0;
    PetscCall(PetscRandomGetValueReal(rx, &xr));
    PetscCall(PetscRandomGetValueReal(ry, &yr));
    PetscCall(PetscRandomGetValueReal(rz, &zr));
    PetscCall(density(xr, yr, zr, &v, ctx));
    PetscCall(PetscRandomGetValueReal(rv, &vr));
    if (v > vr) {
      x = xr;
      y = yr;
      z = zr;
      Np++;
    }
    coords[ip*NDIM + 0] = x;
    coords[ip*NDIM + 1] = y;
    coords[ip*NDIM + 2] = z;
  }

  // Restore the coordinates array.
  PetscCall(DMSwarmRestoreField(
            swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Destroy the random-number generator.
  PetscCall(PetscRandomDestroy(&random));

  // Update the swarm.
  PetscCall(DMSwarmMigrate(swarm, PETSC_TRUE));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeSwarmCoordinates(Context *ctx)
{
  DM          swarm=ctx->swarm;
  PetscInt    np;
  PetscScalar *coords;
  PetscInt    ip;
  PetscRandom random;
  PetscReal   delta;
  PetscReal   dx, x, dy, y, dz, z;

  PetscFunctionBeginUser;

  // Place an equal number of particles in each cell.
  PetscCall(DMSwarmInsertPointsUsingCellDM(
            swarm, DMSWARMPIC_LAYOUT_REGULAR, NPPCELL));

  // Update the particle DM.
  PetscCall(DMSwarmMigrate(swarm, PETSC_TRUE));

  // Create a random-number generator to nudge particle positions.
  delta = 0.5*PetscMin(ctx->grid.d.x, PetscMin(ctx->grid.d.y, ctx->grid.d.z));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &random));
  PetscCall(PetscRandomSetInterval(random, -delta, +delta));
  PetscCall(PetscRandomSetSeed(random, (unsigned long)ctx->mpi.rank));
  PetscCall(PetscRandomSeed(random));

  // Get a representation of the particle coordinates.
  PetscCall(DMSwarmGetField(
            swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  // Loop over particles and assign positions.
  for (ip=0; ip<np; ip++) {
    PetscCall(PetscRandomGetValueReal(random, &dx));
    PetscCall(PetscRandomGetValueReal(random, &dy));
    PetscCall(PetscRandomGetValueReal(random, &dz));
    x = coords[ip*NDIM + 0] + dx;
    if (x < 0.0) x = 0.0 + 1.0e-12;
    if (x > 1.0) x = 1.0 - 1.0e-12;
    y = coords[ip*NDIM + 1] + dy;
    if (y < 0.0) y = 0.0 + 1.0e-12;
    if (y > 1.0) y = 1.0 - 1.0e-12;
    z = coords[ip*NDIM + 2] + dz;
    if (z < 0.0) z = 0.0 + 1.0e-12;
    if (z > 1.0) z = 1.0 - 1.0e-12;
    coords[ip*NDIM + 0] = x;
    coords[ip*NDIM + 1] = y;
    coords[ip*NDIM + 2] = z;
  }

  // Restore the coordinates array.
  PetscCall(DMSwarmRestoreField(
            swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Destroy the random-number generator.
  PetscCall(PetscRandomDestroy(&random));

  // Update the swarm.
  PetscCall(DMSwarmMigrate(swarm, PETSC_TRUE));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeParticles(Context *ctx)
{
  DM          swarm=ctx->swarm;
  PetscInt    np;
  PetscScalar *coords;
  RealVector  *pos, *vel;
  PetscInt    ip;

  PetscFunctionBeginUser;

  // Initialize coordinates in the particle DM.
  PetscCall(InitializeSwarmCoordinates(&(*ctx)));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  // Get an array representation of the swarm coordinates.
  PetscCall(DMSwarmGetField(
            swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Get an array representation of the particle positions.
  PetscCall(DMSwarmGetField(
            swarm,
            "position", NULL, NULL,
            (void **)&pos));

  // Get an array representation of the particle velocities.
  PetscCall(DMSwarmGetField(
            swarm,
            "velocity", NULL, NULL,
            (void **)&vel));

  // Loop over particles and assign parameter values.
  for (ip=0; ip<np; ip++) {
    pos[ip].x = coords[ip*NDIM + 0];
    pos[ip].y = coords[ip*NDIM + 1];
    pos[ip].z = coords[ip*NDIM + 2];
    vel[ip].x = ctx->ions.v0.x;
    vel[ip].y = ctx->ions.v0.y;
    vel[ip].z = ctx->ions.v0.z;
  }

  // Restore the particle-positions array.
  PetscCall(DMSwarmRestoreField(
            swarm,
            "position", NULL, NULL,
            (void **)&pos));

  // Restore the particle-velocities array.
  PetscCall(DMSwarmRestoreField(
            swarm,
            "velocity", NULL, NULL,
            (void **)&vel));

  // Restore the swarm-coordinates array.
  PetscCall(DMSwarmRestoreField(
            swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Display information about the particle DM.
  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));

  // Update the parameter context.
  PetscCall(DMSwarmGetSize(swarm, &ctx->plasma.Np));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
EchoSetup(Context ctx)
{
  PetscFunctionBeginUser;

  PetscCall(PetscOptionsView(NULL, ctx.optionsView));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "\nParameter Values\n"));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView,   "----------------\n"));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Nx = %d\n", ctx.grid.N.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Ny = %d\n", ctx.grid.N.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Nz = %d\n", ctx.grid.N.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "x0 = %f [m]\n", ctx.grid.p0.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "y0 = %f [m]\n", ctx.grid.p0.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "z0 = %f [m]\n", ctx.grid.p0.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "x1 = %f [m]\n", ctx.grid.p1.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "y1 = %f [m]\n", ctx.grid.p1.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "z1 = %f [m]\n", ctx.grid.p1.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Lx = %f [m]\n", ctx.grid.L.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Ly = %f [m]\n", ctx.grid.L.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Lz = %f [m]\n", ctx.grid.L.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "dx = %f [m]\n", ctx.grid.d.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "dy = %f [m]\n", ctx.grid.d.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "dz = %f [m]\n", ctx.grid.d.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Np = %d\n", ctx.plasma.Np));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "B0x = %g [T]\n", ctx.plasma.B0.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "B0y = %g [T]\n", ctx.plasma.B0.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "B0z = %g [T]\n", ctx.plasma.B0.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "E0x = %g [N/C]\n", ctx.plasma.E0.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "E0y = %g [N/C]\n", ctx.plasma.E0.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "E0z = %g [N/C]\n", ctx.plasma.E0.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "nue = %g [s^-1]\n", ctx.electrons.nu));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "ve0x = %f [m/s]\n", ctx.electrons.v0.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "ve0y = %f [m/s]\n", ctx.electrons.v0.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "ve0z = %f [m/s]\n", ctx.electrons.v0.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "veTx = %f [m/s]\n", ctx.electrons.vT.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "veTy = %f [m/s]\n", ctx.electrons.vT.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "veTz = %f [m/s]\n", ctx.electrons.vT.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Te = %f [K]\n", ctx.electrons.T));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Omega_ex = %g [s^-1]\n", ctx.electrons.Omega.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Omega_ey = %g [s^-1]\n", ctx.electrons.Omega.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Omega_ez = %g [s^-1]\n", ctx.electrons.Omega.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "kappa_ex = %g\n", ctx.electrons.kappa.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "kappa_ey = %g\n", ctx.electrons.kappa.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "kappa_ez = %g\n", ctx.electrons.kappa.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "qi = %e [C]\n", ctx.ions.q));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "mi = %e [kg]\n", ctx.ions.m));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "nui = %g [s^-1]\n", ctx.ions.nu));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "vi0x = %f [m/s]\n", ctx.ions.v0.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "vi0y = %f [m/s]\n", ctx.ions.v0.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "vi0z = %f [m/s]\n", ctx.ions.v0.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "viTx = %f [m/s]\n", ctx.ions.vT.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "viTy = %f [m/s]\n", ctx.ions.vT.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "viTz = %f [m/s]\n", ctx.ions.vT.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Ti = %f [K]\n", ctx.ions.T));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Omega_ix = %g [s^-1]\n", ctx.ions.Omega.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Omega_iy = %g [s^-1]\n", ctx.ions.Omega.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "Omega_iz = %g [s^-1]\n", ctx.ions.Omega.z));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "kappa_ix = %g\n", ctx.ions.kappa.x));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "kappa_iy = %g\n", ctx.ions.kappa.y));
  PetscCall(PetscViewerASCIIPrintf(
            ctx.optionsView, "kappa_iz = %g\n", ctx.ions.kappa.z));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CollectParticles(Context *ctx)
{
  DM          grid, swarm=ctx->swarm;
  Vec         gridvec;
  GridNode    ***array;
  PetscInt    dim;
  PetscInt    i0, j0, k0;
  PetscReal   x, y, z, dx, dy, dz;
  RealVector  r, *pos, v, *vel;
  PetscInt    ip, np;
  PetscInt    ixl, ixh, iyl, iyh, izl, izh;
  PetscReal   wxl, wxh, wyl, wyh, wzl, wzh;
  PetscReal   hhh, lhh, hlh, llh, hhl, lhl, hll, lll;
  PetscReal   w[NDIM];

  PetscFunctionBeginUser;
  PUSH_FUNC;

  // Get the grid DM from the swarm DM.
  PetscCall(DMSwarmGetCellDM(swarm, &grid));

  // Get a vector for the local portion of the grid.
  PetscCall(DMGetLocalVector(grid, &gridvec));

  // Make sure the local grid vector has zeroes everywhere.
  PetscCall(VecZeroEntries(gridvec));

  // Get a 4-D array corresponding to the local grid quantities.
  PetscCall(DMDAVecGetArray(grid, gridvec, &array));

  // Get an array representation of the particle positions.
  PetscCall(DMSwarmGetField(
            swarm,
            "position", NULL, NULL,
            (void **)&pos));

  // Get an array representation of the particle velocities.
  PetscCall(DMSwarmGetField(
            swarm,
            "velocity", NULL, NULL,
            (void **)&vel));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  // Compute grid grid spacing.
  // - TODO: This can use ctx->grid.d.{x,y,z}
  PetscCall(DMDAGetInfo(
            grid, NULL,
            &i0, &j0, &k0,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  dx = ctx->grid.L.x / (PetscReal)i0;
  dy = ctx->grid.L.y / (PetscReal)j0;
  dz = ctx->grid.L.z / (PetscReal)k0;

  // Loop over particles.
  for (ip=0; ip<np; ip++) {
    // Get the current particle's parameters.
    r = pos[ip];
    x = r.x / dx;
    y = r.y / dy;
    z = r.z / dz;
    // Compute the x-dimension neighbors and corresponding weights.
    ixl = (PetscInt)x;
    ixh = ixl+1;
    wxh = x - ixl;
    wxl = 1.0 - wxh;
    // Compute the y-dimension neighbors and corresponding weights.
    iyl = (PetscInt)y;
    iyh = iyl+1;
    wyh = y - iyl;
    wyl = 1.0 - wyh;
    // Compute the z-dimension neighbors and corresponding weights.
    izl = (PetscInt)z;
    izh = izl+1;
    wzh = z - izl;
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
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Normalized positions: (%3.1f, %3.1f, %3.1f)\n", r.x, r.y, r.z));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Physical positions:   (%3.1f, %3.1f, %3.1f)\n", x, y, z));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Lower indices:        (%03d, %03d, %03d)\n", ixl, iyl, izl));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Upper indices:        (%03d, %03d, %03d)\n", ixh, iyh, izh));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "               density(%03d, %03d, %03d) + %3.1f = %3.1f\n", ixh, iyh, izh, hhh, array[izh][iyh][ixh].n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "               density(%03d, %03d, %03d) + %3.1f = %3.1f\n", ixh, iyh, izl, lhh, array[izl][iyh][ixh].n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "               density(%03d, %03d, %03d) + %3.1f = %3.1f\n", ixh, iyl, izh, hlh, array[izh][iyl][ixh].n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "               density(%03d, %03d, %03d) + %3.1f = %3.1f\n", ixh, iyl, izl, llh, array[izl][iyl][ixh].n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "               density(%03d, %03d, %03d) + %3.1f = %3.1f\n", ixl, iyh, izh, hhl, array[izh][iyh][ixl].n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "               density(%03d, %03d, %03d) + %3.1f = %3.1f\n", ixl, iyh, izl, lhl, array[izl][iyh][ixl].n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "               density(%03d, %03d, %03d) + %3.1f = %3.1f\n", ixl, iyl, izh, hll, array[izh][iyl][ixl].n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "               density(%03d, %03d, %03d) + %3.1f = %3.1f\n", ixl, iyl, izl, lll, array[izl][iyl][ixl].n));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
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
  PetscCall(DMLocalToGlobal(grid, gridvec, ADD_VALUES, ctx->global));

  // Restore the local grid vector.
  PetscCall(DMRestoreLocalVector(grid, &gridvec));

  // Restore the particle-positions array.
  PetscCall(DMSwarmRestoreField(
            swarm,
            "position", NULL, NULL,
            (void **)&pos));

  // Restore the particle-velocities array.
  PetscCall(DMSwarmRestoreField(
            swarm,
            "velocity", NULL, NULL,
            (void **)&vel));

  POP_FUNC;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Get the global vector corresponding to a named DM field.

  Note: Something like this may exist in PETSc but I can't find it.
*/
static PetscErrorCode
GetFieldVec(DM dm, Vec full, const char *name, Vec *vec)
{
  PetscInt  nf;
  char      **names;
  DM        *dmArray;
  PetscInt  field;
  PetscBool found;

  PetscFunctionBeginUser;

  PetscCall(DMCreateFieldDecomposition(dm, &nf, &names, NULL, &dmArray));
  for (field=0; field<nf; field++) {
    PetscCall(PetscStrcmp(name, names[field], &found));
    if (found) {
      PetscCall(DMGetGlobalVector(dmArray[field], vec));
      PetscCall(VecStrideGather(full, field, *vec, INSERT_VALUES));
      PetscCall(PetscObjectSetName((PetscObject)*vec, names[field]));
      break;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Restore a vector retrieved via GetFieldVec.

  Note: Something like this may exist in PETSc but I can't find it.
*/
static PetscErrorCode
RestoreFieldVec(DM dm, Vec full, const char *name, Vec *vec)
{
  PetscInt  nf;
  char      **names;
  DM        *dmArray;
  PetscInt  field;
  PetscBool found;

  PetscFunctionBeginUser;

  PetscCall(DMCreateFieldDecomposition(dm, &nf, &names, NULL, &dmArray));
  for (field=0; field<nf; field++) {
    PetscCall(PetscStrcmp(name, names[field], &found));
    if (found) {
      PetscCall(DMRestoreGlobalVector(dmArray[field], vec));
      break;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
VecViewComposite(DM grid, Vec full, PetscViewer viewer)
{
  PetscInt nf;
  char **names;
  IS *isArray;
  DM *dmArray;
  PetscInt field;
  Vec vec;

  PetscFunctionBeginUser;

  PetscCall(DMCreateFieldDecomposition(
            grid, &nf, &names, &isArray, &dmArray));
  for (field=0; field<nf; field++) {
    PetscCall(DMGetGlobalVector(dmArray[field], &vec));
    PetscCall(VecStrideGather(full, field, vec, INSERT_VALUES));
    PetscCall(PetscObjectSetName((PetscObject)vec, names[field]));
    PetscCall(VecView(vec, viewer));
    PetscCall(DMRestoreGlobalVector(dmArray[field], &vec));
  }

  for (field=0; field<nf; field++) {
    PetscFree(names[field]);
    PetscCall(ISDestroy(&isArray[field]));
    PetscCall(DMDestroy(&dmArray[field]));
  }
  PetscFree(names);
  PetscFree(isArray);
  PetscFree(dmArray);

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
ComputeConstantRHS(KSP ksp, Vec b, void *_ctx)
{
  Context      *ctx=(Context *)_ctx;
  PetscScalar  Kx, Ky, Kz;
  PetscReal    detA;
  DM           grid;
  PetscReal    dx=ctx->grid.d.x;
  PetscReal    dy=ctx->grid.d.y;
  PetscReal    dz=ctx->grid.d.z;
  Vec          density;
  PetscScalar  mean, val;
  MatNullSpace nullspace;

  PetscFunctionBeginUser;

  // Extract values of electron magnetization.
  Kx = ctx->electrons.kappa.x;
  Ky = ctx->electrons.kappa.y;
  Kz = ctx->electrons.kappa.z;

  // Compute the value of the matrix determinant.
  detA = 1 + Kx*Kx + Ky*Ky + Kz*Kz;

  // Get the grid DM from the context.
  PetscCall(DMSwarmGetCellDM(ctx->swarm, &grid));

  // Extract the density vector.
  PetscCall(GetFieldVec(grid, ctx->global, "density", &density));

  // Set the RHS vector equal to the global mean density.
  PetscCall(VecMean(density, &mean));
  val = mean * (2.0 * dx*dy*dz / detA);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mean = %f\n", mean));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "val = %f\n", val));
  PetscCall(VecSet(b, val));

  // Restore the density vector.
  PetscCall(RestoreFieldVec(grid, ctx->global, "density", &density));

  // Write the RHS vector to HDF5.
  PetscCall(PetscObjectSetName((PetscObject)b, "rhs"));
  PetscCall(VecView(b, ctx->gridView));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ComputeSinusoidalRHS(KSP ksp, Vec b, void *_ctx)
{
  Context      *ctx=(Context *)_ctx;
  PetscScalar  Kx, Ky, Kz;
  PetscReal    detA;
  PetscReal    dx=ctx->grid.d.x;
  PetscReal    dy=ctx->grid.d.y;
  PetscReal    dz=ctx->grid.d.z;
  DM           grid;
  Vec          gridvec;
  GridNode     ***gridarr;
  DM           dm;
  PetscScalar  ***rhs;
  PetscInt     i0, j0, k0;
  PetscInt     ni, nj, nk;
  PetscInt     i, j, k;
  PetscScalar  nijk;
  PetscScalar  x, y, z;
  PetscScalar  Cx, Cy, Cz;
  PetscScalar  val;
  MatNullSpace nullspace;

  PetscFunctionBeginUser;

  // Extract values of electron magnetization.
  Kx = ctx->electrons.kappa.x;
  Ky = ctx->electrons.kappa.y;
  Kz = ctx->electrons.kappa.z;

  // Compute the value of the matrix determinant.
  detA = 1 + Kx*Kx + Ky*Ky + Kz*Kz;

  // Get the grid DM from the context.
  PetscCall(DMSwarmGetCellDM(ctx->swarm, &grid));

  // Extract the density array.
  PetscCall(DMGetLocalVector(grid, &gridvec));
  PetscCall(DMGlobalToLocalBegin(grid, ctx->global, INSERT_VALUES, gridvec));
  PetscCall(DMGlobalToLocalEnd(grid, ctx->global, INSERT_VALUES, gridvec));
  PetscCall(DMDAVecGetArray(grid, gridvec, &gridarr));

  // Zero the incoming vector.
  PetscCall(VecZeroEntries(b));

  // Get the DM associated with the KSP.
  PetscCall(KSPGetDM(ksp, &dm));

  // Get an array equivalent to the RHS Vec.
  PetscCall(DMDAVecGetArray(dm, b, &rhs));

  // Get this processor's indices.
  PetscCall(DMDAGetCorners(dm, &i0, &j0, &k0, &ni, &nj, &nk));

  // Loop over grid points.
  for (k=k0; k<k0+nk; k++) {
    for (j=j0; j<j0+nj; j++) {
      for (i=i0; i<i0+ni; i++) {
        nijk = gridarr[k][j][i].n;
        x = ((PetscReal)i + 0.5)*dx;
        y = ((PetscReal)j + 0.5)*dy;
        z = ((PetscReal)k + 0.5)*dz;
        Cx = PetscCosScalar(2*PETSC_PI * x);
        Cy = PetscCosScalar(2*PETSC_PI * y);
        Cz = PetscCosScalar(2*PETSC_PI * z);
        val = Cx * Cy * Cz;
        rhs[k][j][i] = val * (2.0 * dx*dy*dz / detA);
      }
    }
  }

  // Restore the borrowed arrays.
  PetscCall(DMDAVecRestoreArray(grid, gridvec, &gridarr));
  PetscCall(DMDAVecRestoreArray(dm, b, &rhs));

  // Make the RHS vector consistent with the LHS operator.
  PetscCall(MatNullSpaceCreate(
            PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  // Write the RHS vector to HDF5.
  PetscCall(PetscObjectSetName((PetscObject)b, "rhs"));
  PetscCall(VecView(b, ctx->gridView));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ComputeFullRHS(KSP ksp, Vec b, void *_ctx)
{
  // the problem context
  Context      *ctx=(Context *)_ctx;
  // components of magnetization vector
  PetscScalar  Kx, Ky, Kz;
  // matrix determinant
  PetscReal    detA;
  // components of magnetization tensor
  PetscScalar  rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz;
  // x-axis cell spacing
  PetscReal    dx=ctx->grid.d.x;
  // y-axis cell spacing
  PetscReal    dy=ctx->grid.d.y;
  // z-axis cell spacing
  PetscReal    dz=ctx->grid.d.z;
  // the DM of the grid
  DM           grid;
  // local grid vector
  Vec          gridvec;
  // array representation of grid quantities
  GridNode     ***gridarr;
  // the DM of the KSP
  DM           dm;
  // array of RHS values
  PetscScalar  ***rhs;
  // indices of the lower left corner of the local grid
  PetscInt     i0, j0, k0;
  // number of non-ghost cells in each dimension of the local grid
  PetscInt     ni, nj, nk;
  // grid indices
  PetscInt     i, j, k;
  // the density value at the current and neighboring grid points
  PetscScalar  nijk, npjk, nmjk, nipk, nimk, nijp, nijm;
  PetscScalar  nppk, npmk, nmpk, nmmk;
  PetscScalar  npjp, npjm, nmjp, nmjm;
  PetscScalar  nipp, nipm, nimp, nimm;
  PetscScalar  E0x=ctx->plasma.E0.x;
  PetscScalar  E0y=ctx->plasma.E0.y;
  PetscScalar  E0z=ctx->plasma.E0.z;
  PetscScalar  cth=ctx->electrons.gamma*KB*ctx->electrons.T/Q;
  PetscScalar  cG=ctx->electrons.m*ctx->electrons.nu/Q;
  PetscScalar  val;
  // the operator nullspace
  MatNullSpace nullspace;

  PetscFunctionBeginUser;

  // Extract values of electron magnetization.
  Kx = ctx->electrons.kappa.x;
  Ky = ctx->electrons.kappa.y;
  Kz = ctx->electrons.kappa.z;

  // Compute the value of the matrix determinant.
  detA = 1 + Kx*Kx + Ky*Ky + Kz*Kz;

  // Define components of the magnetization tensor.
  rxx = 1 + Kx*Kx;
  rxy = Ky*Kx - Kz;
  rxz = Kz*Kx + Ky;
  ryx = Kx*Ky + Kz;
  ryy = 1 + Ky*Ky;
  ryz = Kz*Ky - Kx;
  rzx = Kx*Kz - Ky;
  rzy = Ky*Kz + Kx;
  rzz = 1 + Kz*Kz;

  // Get the grid DM from the context.
  PetscCall(DMSwarmGetCellDM(ctx->swarm, &grid));

  // Extract the density array.
  PetscCall(DMGetLocalVector(grid, &gridvec));
  PetscCall(DMGlobalToLocalBegin(grid, ctx->global, INSERT_VALUES, gridvec));
  PetscCall(DMGlobalToLocalEnd(grid, ctx->global, INSERT_VALUES, gridvec));
  PetscCall(DMDAVecGetArray(grid, gridvec, &gridarr));

  // Zero the incoming vector.
  PetscCall(VecZeroEntries(b));

  // Get the DM associated with the KSP.
  PetscCall(KSPGetDM(ksp, &dm));

  // Get an array equivalent to the RHS Vec.
  PetscCall(DMDAVecGetArray(dm, b, &rhs));

  // Get this processor's indices.
  PetscCall(DMDAGetCorners(dm, &i0, &j0, &k0, &ni, &nj, &nk));

  // Loop over grid points.
  for (k=k0; k<k0+nk; k++) {
    for (j=j0; j<j0+nj; j++) {
      for (i=i0; i<i0+ni; i++) {

        // Assign temporary density values.
        nijk = gridarr[k][j][i].n;
        nmjk = gridarr[k][j][i-1].n;
        npjk = gridarr[k][j][i+1].n;
        nimk = gridarr[k][j-1][i].n;
        nipk = gridarr[k][j+1][i].n;
        nijm = gridarr[k-1][j][i].n;
        nijp = gridarr[k+1][j][i].n;
        nppk = gridarr[k][j+1][i+1].n;
        npmk = gridarr[k][j-1][i+1].n;
        nmpk = gridarr[k][j+1][i-1].n;
        nmmk = gridarr[k][j-1][i-1].n;
        npjp = gridarr[k+1][j][i+1].n;
        npjm = gridarr[k-1][j][i+1].n;
        nmjp = gridarr[k+1][j][i-1].n;
        nmjm = gridarr[k-1][j][i-1].n;
        nipp = gridarr[k+1][j+1][i].n;
        nipm = gridarr[k-1][j+1][i].n;
        nimp = gridarr[k+1][j-1][i].n;
        nimm = gridarr[k-1][j-1][i].n;

        // Assign the RHS value at (i, j, k).
        val =
          // div(n R E0)
          (rxx*E0x + rxy*E0y + rxz*E0z)*(npjk - nmjk)*dy*dz +
          (ryx*E0x + ryy*E0y + ryz*E0z)*(nipk - nimk)*dy*dz +
          (rzx*E0x + rzy*E0y + rzz*E0z)*(nijp - nijm)*dy*dz +
          // div(R div(P)) / e
          cth * (
            rxx * (npjk - 2.0*nijk + nmjk) * (2.0*dy*dz/dx) +
            rxy * (nppk - npmk - nmpk + nmmk) * (0.5*dz) +
            rxz * (npjp - npjm - nmjp + nmjm) * (0.5*dy) +
            ryx * (nppk - npmk - nmpk + nmmk) * (0.5*dz) +
            ryy * (nipk - 2.0*nijk + nimk) * (2.0*dx*dz/dy) +
            ryz * (nipp - nipm - nimp + nimm) * (0.5*dx) +
            rzx * (npjp - npjm - nmjp + nmjm) * (0.5*dy) +
            rzy * (nipp - nipm - nimp + nimm) * (0.5*dx) +
            rzz * (nijp - 2.0*nijk + nijm) * (2.0*dx*dy/dz)) +
          // (1+kappa^2) (me nue / e) div(flux)
          detA * cG * (
            (gridarr[k][j][i+1].flux[0]-gridarr[k][j][i-1].flux[0])*(dy*dz) +
            (gridarr[k][j+1][i].flux[1]-gridarr[k][j-1][i].flux[1])*(dx*dz) +
            (gridarr[k+1][j][i].flux[2]-gridarr[k-1][j][i].flux[2])*(dx*dy));
        rhs[k][j][i] = val / detA;
      }
    }
  }

  // Restore the borrowed arrays.
  PetscCall(DMDAVecRestoreArray(grid, gridvec, &gridarr));
  PetscCall(DMDAVecRestoreArray(dm, b, &rhs));

  // Make the RHS vector consistent with the LHS operator.
  PetscCall(MatNullSpaceCreate(
            PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  // Write the RHS vector to HDF5.
  PetscCall(PetscObjectSetName((PetscObject)b, "rhs"));
  PetscCall(VecView(b, ctx->gridView));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ComputeIdentityLHS(KSP ksp, Mat J, Mat A, void *_ctx)
{
  Context      *ctx=(Context *)_ctx;

  PetscFunctionBeginUser;

  PetscCall(MatZeroEntries(A));
  PetscCall(MatShift(A, 1.0));

  if (ctx->viewLHS) {
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(
              PETSC_COMM_WORLD,
              "lhs.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(MatView(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


// Note: This is more complicated than necessary because it arose as a
// simplification of ComputeFullLHS. It could be further simplified.
static PetscErrorCode
ComputeLaplacianLHS(KSP ksp, Mat J, Mat A, void *_ctx)
{
  // the problem context
  Context      *ctx=(Context *)_ctx;
  // geometric scale factors
  PetscScalar  sxx, syx, szx, sxy, syy, szy, sxz, syz, szz;
  // the DM of the KSP
  DM           dm;
  // indices of the lower left corner of the local grid
  PetscInt     i0, j0, k0;
  // number of non-ghost cells in each dimension of the local grid
  PetscInt     ni, nj, nk;
  // grid indices
  PetscInt     i, j, k;
  // the density value at the current and neighboring grid points
  PetscScalar  nijk, npjk, nmjk, nipk, nimk, nijp, nijm;
  // diagonal coefficient
  PetscScalar  vijk=1.0;
  // star-stencil coefficients
  PetscScalar  vpjk=0.0, vmjk=0.0, vipk=0.0, vimk=0.0, vijp=0.0, vijm=0.0;
  // x-y corners
  PetscScalar  vmmk=0.0, vpmk=0.0, vmpk=0.0, vppk=0.0;
  // x-z corners
  PetscScalar  vpjp=0.0, vpjm=0.0, vmjp=0.0, vmjm=0.0;
  // y-z corners
  PetscScalar  vipp=0.0, vipm=0.0, vimp=0.0, vimm=0.0;
  // the current value at each active stencil point
  PetscScalar  val[NVALUES];
  // the current matrix row
  MatStencil   row;
  // the current matrix column of each active stencil point
  MatStencil   col[NVALUES];
  // the operator nullspace
  MatNullSpace nullspace;

  PetscFunctionBeginUser;

  // Compute constant stencil values.
  sxx = ctx->grid.d.y * ctx->grid.d.z / ctx->grid.d.x;
  syx = 0.25*ctx->grid.d.z;
  szx = 0.25*ctx->grid.d.y;
  sxy = 0.25*ctx->grid.d.z;
  syy = ctx->grid.d.x * ctx->grid.d.z / ctx->grid.d.y;
  szy = 0.25*ctx->grid.d.x;
  sxz = 0.25*ctx->grid.d.y;
  syz = 0.25*ctx->grid.d.x;
  szz = ctx->grid.d.x * ctx->grid.d.y / ctx->grid.d.z;

  // Assign the star-stencil coefficients.
  vpjk =  sxx;
  vmjk =  sxx;
  vipk =  syy;
  vimk =  syy;
  vijp =  szz;
  vijm =  szz;

  // Assign the diagonal coefficient.
  vijk = -(vpjk + vipk + vijp + vmjk + vimk + vijm);

  // Get the DM associated with the KSP.
  PetscCall(KSPGetDM(ksp, &dm));

  // Get this processor's indices.
  PetscCall(DMDAGetCorners(dm, &i0, &j0, &k0, &ni, &nj, &nk));

  // Loop over grid points.
  for (k=k0; k<k0+nk; k++) {
    for (j=j0; j<j0+nj; j++) {
      for (i=i0; i<i0+ni; i++) {
        row.i = i; row.j = j; row.k = k;
        // Interior node (i+1, j, k)
        val[0] = vpjk;
        col[0].i = i+1;
        col[0].j = j;
        col[0].k = k;
        // Interior node (i-1, j, k)
        val[1] = vmjk;
        col[1].i = i-1;
        col[1].j = j;
        col[1].k = k;
        // Interior node (i, j+1, k)
        val[2] = vipk;
        col[2].i = i;
        col[2].j = j+1;
        col[2].k = k;
        // Interior node (i, j-1, k)
        val[3] = vimk;
        col[3].i = i;
        col[3].j = j-1;
        col[3].k = k;
        // Interior node (i, j, k+1)
        val[4] = vijp;
        col[4].i = i;
        col[4].j = j;
        col[4].k = k+1;
        // Interior node (i, j, k-1)
        val[5] = vijm;
        col[5].i = i;
        col[5].j = j;
        col[5].k = k-1;
        // Interior node (i+1, j+1, k)
        val[6] = vppk;
        col[6].i = i+1;
        col[6].j = j+1;
        col[6].k = k;
        // Interior node (i+1, j-1, k)
        val[7] = vpmk;
        col[7].i = i+1;
        col[7].j = j-1;
        col[7].k = k;
        // Interior node (i-1, j+1, k)
        val[8] = vmpk;
        col[8].i = i-1;
        col[8].j = j+1;
        col[8].k = k;
        // Interior node (i-1, j-1, k)
        val[9] = vmmk;
        col[9].i = i-1;
        col[9].j = j-1;
        col[9].k = k;
        // Interior node (i+1, j, k+1)
        val[10] = vpjp;
        col[10].i = i+1;
        col[10].j = j;
        col[10].k = k+1;
        // Interior node (i+1, j, k-1)
        val[11] = vpjm;
        col[11].i = i+1;
        col[11].j = j;
        col[11].k = k-1;
        // Interior node (i-1, j, k+1)
        val[12] = vmjp;
        col[12].i = i-1;
        col[12].j = j;
        col[12].k = k+1;
        // Interior node (i-1, j, k-1)
        val[13] = vmjm;
        col[13].i = i-1;
        col[13].j = j;
        col[13].k = k-1;
        // Interior node (i, j+1, k+1)
        val[14] = vipp;
        col[14].i = i;
        col[14].j = j+1;
        col[14].k = k+1;
        // Interior node (i, j+1, k-1)
        val[15] = vipm;
        col[15].i = i;
        col[15].j = j+1;
        col[15].k = k-1;
        // Interior node (i, j-1, k+1)
        val[16] = vimp;
        col[16].i = i;
        col[16].j = j-1;
        col[16].k = k+1;
        // Interior node (i, j-1, k-1)
        val[17] = vimm;
        col[17].i = i;
        col[17].j = j-1;
        col[17].k = k-1;
        // Interior node (i, j, k)
        val[18] = vijk;
        col[18].i = row.i;
        col[18].j = row.j;
        col[18].k = row.k;
        PetscCall(MatSetValuesStencil(
                  A, 1, &row, NVALUES, col, val, INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatNullSpaceCreate(
            PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(A, nullspace));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  if (ctx->viewLHS) {
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(
              PETSC_COMM_WORLD,
              "lhs.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(MatView(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ComputeFullLHS(KSP ksp, Mat J, Mat A, void *_ctx)
{
  // the problem context
  Context      *ctx=(Context *)_ctx;
  // components of magnetization vector
  PetscScalar  Kx, Ky, Kz;
  // matrix determinant
  PetscReal    detA;
  // components of magnetization tensor
  PetscScalar  rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz;
  // geometric scale factors
  PetscScalar  sxx, syx, szx, sxy, syy, szy, sxz, syz, szz;
  // the DM of the grid
  DM           grid;
  // local grid vector
  Vec          gridvec;
  // array representation of grid quantities
  GridNode  ***array;
  // the DM of the KSP
  DM           dm;
  // indices of the lower left corner of the local grid
  PetscInt     i0, j0, k0;
  // number of non-ghost cells in each dimension of the local grid
  PetscInt     ni, nj, nk;
  // grid indices
  PetscInt     i, j, k;
  // the density value at the current and neighboring grid points
  PetscScalar  nijk, npjk, nmjk, nipk, nimk, nijp, nijm;
  // diagonal coefficient
  PetscScalar  vijk=1.0;
  // star-stencil coefficients
  PetscScalar  vpjk=0.0, vmjk=0.0, vipk=0.0, vimk=0.0, vijp=0.0, vijm=0.0;
  // x-y corners
  PetscScalar  vmmk=0.0, vpmk=0.0, vmpk=0.0, vppk=0.0;
  // x-z corners
  PetscScalar  vpjp=0.0, vpjm=0.0, vmjp=0.0, vmjm=0.0;
  // y-z corners
  PetscScalar  vipp=0.0, vipm=0.0, vimp=0.0, vimm=0.0;
  // the current value at each active stencil point
  PetscScalar  val[NVALUES];
  // the current matrix row
  MatStencil   row;
  // the current matrix column of each active stencil point
  MatStencil   col[NVALUES];
  // the operator nullspace
  MatNullSpace nullspace;

  PetscFunctionBeginUser;

  // Extract values of electron magnetization.
  Kx = ctx->electrons.kappa.x;
  Ky = ctx->electrons.kappa.y;
  Kz = ctx->electrons.kappa.z;

  // Compute the value of the matrix determinant.
  detA = 1 + Kx*Kx + Ky*Ky + Kz*Kz;

  // Define components of the magnetization tensor.
  rxx = 1 + Kx*Kx;
  rxy = Ky*Kx - Kz;
  rxz = Kz*Kx + Ky;
  ryx = Kx*Ky + Kz;
  ryy = 1 + Ky*Ky;
  ryz = Kz*Ky - Kx;
  rzx = Kx*Kz - Ky;
  rzy = Ky*Kz + Kx;
  rzz = 1 + Kz*Kz;

  /* Compute geometric scale factors for stencil values. Note that there is some
  redundancy for the sake of organization.

  Diagonal factors have the form
  - `sii = 2*di*dj*dk / (2*di^2*|A|) = dj*dk / (di*|A|)`

  Off-diagonal factors have the form
  - `sij = 2*di*dj*dk / (8*di*dj*|A|) = dk / (4*|A|)`
  */
  sxx = ctx->grid.d.y * ctx->grid.d.z / ctx->grid.d.x / detA;
  syx = 0.25*ctx->grid.d.z  / detA;
  szx = 0.25*ctx->grid.d.y  / detA;
  sxy = 0.25*ctx->grid.d.z  / detA;
  syy = ctx->grid.d.x * ctx->grid.d.z / ctx->grid.d.y / detA;
  szy = 0.25*ctx->grid.d.x  / detA;
  sxz = 0.25*ctx->grid.d.y  / detA;
  syz = 0.25*ctx->grid.d.x  / detA;
  szz = ctx->grid.d.x * ctx->grid.d.y / ctx->grid.d.z / detA;

  // Get the grid DM from the context.
  PetscCall(DMSwarmGetCellDM(ctx->swarm, &grid));

  // Extract the density array.
  PetscCall(DMGetLocalVector(grid, &gridvec));
  PetscCall(DMGlobalToLocalBegin(grid, ctx->global, INSERT_VALUES, gridvec));
  PetscCall(DMGlobalToLocalEnd(grid, ctx->global, INSERT_VALUES, gridvec));
  PetscCall(DMDAVecGetArray(grid, gridvec, &array));

  // Get the DM associated with the KSP.
  PetscCall(KSPGetDM(ksp, &dm));

  // Get this processor's indices.
  PetscCall(DMDAGetCorners(dm, &i0, &j0, &k0, &ni, &nj, &nk));

  // Loop over grid points.
  for (k=k0; k<k0+nk; k++) {
    for (j=j0; j<j0+nj; j++) {
      for (i=i0; i<i0+ni; i++) {

        // Assign density values
        nijk = array[k][j][i].n;
        nmjk = array[k][j][i-1].n;
        npjk = array[k][j][i+1].n;
        nimk = array[k][j-1][i].n;
        nipk = array[k][j+1][i].n;
        nijm = array[k-1][j][i].n;
        nijp = array[k+1][j][i].n;

        /* x-y corner coefficients */
        vppk =  sxy*rxy*(npjk + nijk) + syx*ryx*(nipk + nijk);
        vpmk = -sxy*rxy*(npjk + nijk) - syx*ryx*(nijk + nimk);
        vmpk = -sxy*rxy*(nijk + nmjk) - syx*ryx*(nipk + nijk);
        vmmk =  sxy*rxy*(nijk + nmjk) + syx*ryx*(nijk + nimk);
        /* x-z corner coefficients */
        vpjp =  sxz*rxz*(npjk + nijk) + szx*rzx*(nijp + nijk);
        vpjm = -sxz*rxz*(npjk + nijk) - szx*rzx*(nijk + nijm);
        vmjp = -sxz*rxz*(nijk + nmjk) - szx*rzx*(nijp + nijk);
        vmjm =  sxz*rxz*(nijk + nmjk) + szx*rzx*(nijk + nijm);
        /* y-z corner coefficients */
        vipp =  syz*ryz*(nipk + nijk) + szy*rzy*(nijp + nijk);
        vipm = -syz*ryz*(nipk + nijk) - szy*rzy*(nijk + nijm);
        vimp = -syz*ryz*(nijk + nimk) - szy*rzy*(nijp + nijk);
        vimm =  syz*ryz*(nijk + nimk) + szy*rzy*(nijk + nijm);
        /* star-stencil coefficients */
        vpjk =  sxx*rxx*(npjk + nijk) + vppk + vpmk + vpjp + vpjm;
        vmjk =  sxx*rxx*(nijk + nmjk) + vmpk + vmmk + vmjp + vmjm;
        vipk =  syy*ryy*(nipk + nijk) + vppk + vmpk + vipp + vipm;
        vimk =  syy*ryy*(nijk + nimk) + vpmk + vmmk + vimp + vimm;
        vijp =  szz*rzz*(nijp + nijk) + vpjp + vmjp + vipp + vimp;
        vijm =  szz*rzz*(nijk + nijm) + vpjm + vmjm + vipm + vimm;
        /* diagonal coefficient */
        vijk = -(vpjk + vipk + vijp + vmjk + vimk + vijm);

        row.i = i; row.j = j; row.k = k;
        // Interior node (i+1, j, k)
        val[0] = vpjk;
        col[0].i = i+1;
        col[0].j = j;
        col[0].k = k;
        // Interior node (i-1, j, k)
        val[1] = vmjk;
        col[1].i = i-1;
        col[1].j = j;
        col[1].k = k;
        // Interior node (i, j+1, k)
        val[2] = vipk;
        col[2].i = i;
        col[2].j = j+1;
        col[2].k = k;
        // Interior node (i, j-1, k)
        val[3] = vimk;
        col[3].i = i;
        col[3].j = j-1;
        col[3].k = k;
        // Interior node (i, j, k+1)
        val[4] = vijp;
        col[4].i = i;
        col[4].j = j;
        col[4].k = k+1;
        // Interior node (i, j, k-1)
        val[5] = vijm;
        col[5].i = i;
        col[5].j = j;
        col[5].k = k-1;
        // Interior node (i+1, j+1, k)
        val[6] = vppk;
        col[6].i = i+1;
        col[6].j = j+1;
        col[6].k = k;
        // Interior node (i+1, j-1, k)
        val[7] = vpmk;
        col[7].i = i+1;
        col[7].j = j-1;
        col[7].k = k;
        // Interior node (i-1, j+1, k)
        val[8] = vmpk;
        col[8].i = i-1;
        col[8].j = j+1;
        col[8].k = k;
        // Interior node (i-1, j-1, k)
        val[9] = vmmk;
        col[9].i = i-1;
        col[9].j = j-1;
        col[9].k = k;
        // Interior node (i+1, j, k+1)
        val[10] = vpjp;
        col[10].i = i+1;
        col[10].j = j;
        col[10].k = k+1;
        // Interior node (i+1, j, k-1)
        val[11] = vpjm;
        col[11].i = i+1;
        col[11].j = j;
        col[11].k = k-1;
        // Interior node (i-1, j, k+1)
        val[12] = vmjp;
        col[12].i = i-1;
        col[12].j = j;
        col[12].k = k+1;
        // Interior node (i-1, j, k-1)
        val[13] = vmjm;
        col[13].i = i-1;
        col[13].j = j;
        col[13].k = k-1;
        // Interior node (i, j+1, k+1)
        val[14] = vipp;
        col[14].i = i;
        col[14].j = j+1;
        col[14].k = k+1;
        // Interior node (i, j+1, k-1)
        val[15] = vipm;
        col[15].i = i;
        col[15].j = j+1;
        col[15].k = k-1;
        // Interior node (i, j-1, k+1)
        val[16] = vimp;
        col[16].i = i;
        col[16].j = j-1;
        col[16].k = k+1;
        // Interior node (i, j-1, k-1)
        val[17] = vimm;
        col[17].i = i;
        col[17].j = j-1;
        col[17].k = k-1;
        // Interior node (i, j, k)
        val[18] = vijk;
        col[18].i = row.i;
        col[18].j = row.j;
        col[18].k = row.k;
        PetscCall(MatSetValuesStencil(
                  A, 1, &row, NVALUES, col, val, INSERT_VALUES));
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(grid, gridvec, &array));
  PetscCall(DMRestoreLocalVector(grid, &gridvec));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatNullSpaceCreate(
            PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(A, nullspace));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  if (ctx->viewLHS) {
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(
              PETSC_COMM_WORLD,
              "lhs.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(MatView(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  Context      *user=(Context *)ctx;

  PetscFunctionBeginUser;

  switch (user->rhsType) {
  case RHS_CONSTANT:
    PetscCall(ComputeConstantRHS(ksp, b, ctx));
    break;
  case RHS_SINUSOIDAL:
    PetscCall(ComputeSinusoidalRHS(ksp, b, ctx));
    break;
  case RHS_FULL:
    PetscCall(ComputeFullRHS(ksp, b, ctx));
    break;
  default:
    SETERRQ(
      PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "Unknown RHS type: \"%s\"\n", RHSTypes[user->rhsType]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ComputeLHS(KSP ksp, Mat J, Mat A, void *ctx)
{
  Context      *user=(Context *)ctx;

  PetscFunctionBeginUser;

  switch (user->lhsType) {
  case LHS_IDENTITY:
    PetscCall(ComputeIdentityLHS(ksp, J, A, ctx));
    break;
  case LHS_LAPLACIAN:
    PetscCall(ComputeLaplacianLHS(ksp, J, A, ctx));
    break;
  case LHS_FULL:
    PetscCall(ComputeFullLHS(ksp, J, A, ctx));
    break;
  default:
    SETERRQ(
      PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
      "Unknown LHS type: \"%s\"\n", LHSTypes[user->lhsType]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/*
Notes
-----
* This function is supposed to provide a way to view the basic structure of the
  LHS operator matrix, even for production-sized runs, without potentially
  creating a very large binary file.
* It doesn't currently work because the local N{x,y,z} arguments are not
  necessarily consistent with their corresponding values in the application
  context. We could consider defining a new context to pass to this function,
  but that may require refactoring ComputeLHS.
*/
static PetscErrorCode
ViewReducedLHS(PetscInt Nx, PetscInt Ny, PetscInt Nz, void *ctx)
{
  KSP            ksp;
  DM             dm;
  DMBoundaryType xBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType yBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType zBC=DM_BOUNDARY_PERIODIC;
  Mat            A;
  PetscViewer    viewer;

  PetscFunctionBeginUser;

  PetscCall(DMDACreate3d(
            PETSC_COMM_WORLD,
            xBC, yBC, zBC,
            DMDA_STENCIL_BOX,
            Nx, Ny, Nz,
            PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
            1, 1,
            NULL, NULL, NULL,
            &dm));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, dm));
  PetscCall(KSPSetComputeOperators(ksp, ComputeLHS, &ctx));
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(PetscViewerBinaryOpen(
            PETSC_COMM_WORLD,
            "lhs.dat", FILE_MODE_WRITE, &viewer));
  PetscCall(MatView(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&dm));
  PetscCall(KSPDestroy(&ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  MPIContext  mpi;
  time_t      startTime, endTime;
  Context     ctx;
  DM          grid, solve;
  KSP         ksp;
  Vec         x;

  PetscFunctionBeginUser;

  // Initialize PETSc and MPI.
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &mpi.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi.size));
  if (mpi.rank == 0) {
    time(&startTime);
  }
  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "\n**************** START *****************\n\n"));

  // Assign parameter values from user arguments or defaults.
  PetscCall(ProcessOptions(&ctx));

  // Set up the viewer for simulated quantities.
  PetscCall(PetscViewerHDF5Open(
            PETSC_COMM_WORLD, "grid.hdf", FILE_MODE_WRITE,
            &ctx.gridView));

  // Store MPI information in the application context.
  ctx.mpi = mpi;

  // Set up discrete grid.
  PetscCall(InitializeGridDM(&grid, &ctx));
  PetscCall(DMCreateGlobalVector(grid, &ctx.global));
  PetscCall(VecZeroEntries(ctx.global));

  // Set up particle swarm.
  PetscCall(InitializeSwarmDM(grid, &ctx));

  // Set initial particle positions and velocities.
  PetscCall(InitializeParticles(&ctx));

  // Echo the initial state.
  PetscCall(PetscViewerASCIIOpen(
            PETSC_COMM_WORLD, "options.txt", &ctx.optionsView));
  if (mpi.rank == 0) {
    PetscCall(EchoSetup(ctx));
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  // Compute initial density and flux.
  PetscCall(CollectParticles(&ctx));

  // Compute initial electric field.
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(InitializePotentialDM(grid, &solve));
  PetscCall(KSPSetDM(ksp, solve));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetComputeInitialGuess(ksp, ComputeInitialPhi, &ctx));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, &ctx));
  PetscCall(KSPSetComputeOperators(ksp, ComputeLHS, &ctx));
  PetscCall(KSPSolve(ksp, NULL, NULL));
  PetscCall(KSPGetSolution(ksp, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "potential"));

  // Output initial conditions.
  PetscCall(VecViewComposite(grid, ctx.global, ctx.gridView));
  PetscCall(VecView(x, ctx.gridView));

  // Main time-step loop. See KSP ex70.c::SolveTimeDepStokes (~ line 1170) for
  // possible structure of time-step loop.

    // Compute density and flux from particle positions.

    // Compute potential from density

    // Compute electric field from potential

    // Update velocities: $\frac{d\vec{v}}{dt} = \frac{e\vec{E}}{m_i}$. See SNES
    // ex63.c::main (~ line 469) for possible structure.

      // Apply Boris mover

      // Apply collisions

    // Update positions: $\frac{d\vec{r}}{dt} = \vec{v}$

    // Apply boundary conditions

    // Output current time step

  // Free memory.
  PetscCall(PetscViewerDestroy(&ctx.optionsView));
  PetscCall(PetscViewerDestroy(&ctx.gridView));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&ctx.global));
  PetscCall(DMDestroy(&grid));
  PetscCall(DMDestroy(&ctx.swarm));
  PetscCall(DMDestroy(&solve));

  // Write time information
  if (mpi.rank == 0) {
    time(&endTime);
  }
  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "\n----------------------------------------\n"));
  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "Start time: %s", asctime(localtime(&startTime))));
  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "End time:   %s", asctime(localtime(&endTime))));
  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "----------------------------------------\n"));

  // Finalize PETSc and MPI.
  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "\n***************** END ******************\n"));
  PetscCall(PetscFinalize());

  return 0;
}