/* 3D Hybrid PIC */
static char help[] = "A 3D hybrid particle-in-cell (PIC) simulation.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include <petscviewerhdf5.h>


#define NDIM 3
#define Q 1.6022e-19  // fundamental charge in C
#define MP 1.6726e-27 // proton mass in kg
#define ME 9.1094e-13 // electron mass in kg

// Temporary declaration of number of particles per cell, per dimension.
#define NPPCELL 1

// The number of non-null values in the LHS-operator matrix stencil. This is
// equivalent to the standard box stencil with the corners removed.
#define NVALUES 19

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
} Species;

typedef struct {
  RealVector B0; // constant magnetic-field amplitude
  RealVector E0; // constant electric-field amplitude
  PetscReal  Np; // number of charged particles
} Plasma;

typedef struct {
  PetscMPIInt rank; // global processor number
  PetscMPIInt size; // total number of processors
} MPIContext;

typedef struct {
  Grid       grid;      // grid information
  Species    electrons; // electron parameter values
  Species    ions;      // ion parameter values
  Plasma     plasma;    // plasma information
  MPIContext mpi;       // MPI information
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
  PetscBool found;

  // Set fundamental parameter values.
  ctx->electrons.q = -Q;
  ctx->electrons.m = ME;

  // Read optional parameter values from user input.
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
    ctx->electrons.q * ctx->plasma.B0.x / ctx->electrons.m;
  ctx->electrons.Omega.y = 
    ctx->electrons.q * ctx->plasma.B0.y / ctx->electrons.m;
  ctx->electrons.Omega.z = 
    ctx->electrons.q * ctx->plasma.B0.z / ctx->electrons.m;
  ctx->ions.Omega.x = 
    ctx->ions.q * ctx->plasma.B0.x / ctx->ions.m;
  ctx->ions.Omega.y = 
    ctx->ions.q * ctx->plasma.B0.y / ctx->ions.m;
  ctx->ions.Omega.z = 
    ctx->ions.q * ctx->plasma.B0.z / ctx->ions.m;
  // Set species magnetization from Omega and nu.
  ctx->electrons.kappa.x = ctx->electrons.Omega.x / ctx->electrons.nu;
  ctx->electrons.kappa.y = ctx->electrons.Omega.y / ctx->electrons.nu;
  ctx->electrons.kappa.z = ctx->electrons.Omega.z / ctx->electrons.nu;
  ctx->ions.kappa.x = ctx->ions.Omega.x / ctx->ions.nu;
  ctx->ions.kappa.y = ctx->ions.Omega.y / ctx->ions.nu;
  ctx->ions.kappa.z = ctx->ions.Omega.z / ctx->ions.nu;

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
  PetscInt       dof=5;
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

  PetscCall(DMConvert(grid, DMDA, solve));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeSwarmDM(DM *swarm, DM *grid, Context *ctx)
{
  PetscInt dim;
  PetscInt bufsize=0;

  PetscFunctionBeginUser;

  // Create the swarm DM.
  PetscCall(DMCreate(PETSC_COMM_WORLD, swarm));
  // Perform basic setup.
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*swarm), "ions_"));
  PetscCall(DMSetType(*swarm, DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject)*swarm, "Ions"));
  // Synchronize the swarm DM with the grid DM.
  PetscCall(DMGetDimension(*grid, &dim));
  PetscCall(DMSetDimension(*swarm, dim));
  PetscCall(DMSwarmSetCellDM(*swarm, *grid));
  // Declare this to be a PIC swarm. This must occur after setting `dim`.
  PetscCall(DMSwarmSetType(*swarm, DMSWARM_PIC));
  // Register fields that each particle will have.
  PetscCall(DMSwarmInitializeFieldRegister(*swarm));
  PetscCall(DMSwarmRegisterUserStructField(
            *swarm, "position", sizeof(RealVector)));
  PetscCall(DMSwarmRegisterUserStructField(
            *swarm, "velocity", sizeof(RealVector)));
  PetscCall(DMSwarmFinalizeFieldRegister(*swarm));
  // Set the per-processor swarm size and buffer length for efficient resizing.
  PetscCall(DMSwarmSetLocalSizes(
            *swarm, ctx->plasma.Np / ctx->mpi.size, bufsize));
  // View information about the swarm DM.
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeSwarmCoordinates(DM *swarm, Context *ctx)
{
  PetscInt    np;
  PetscScalar *coords;
  PetscInt    ip;
  PetscMPIInt rank;
  PetscRandom random;
  PetscReal   dx, x, dy, y, dz, z;

  PetscFunctionBeginUser;

  // Place an equal number of particles in each cell.
  PetscCall(DMSwarmInsertPointsUsingCellDM(
            *swarm, DMSWARMPIC_LAYOUT_REGULAR, NPPCELL));

  // Update the particle DM.
  PetscCall(DMSwarmMigrate(*swarm, PETSC_TRUE));

  // Create a random-number generator to nudge particle positions.
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &random));
  PetscCall(PetscRandomSetInterval(random, -0.1, +0.1));
  PetscCall(PetscRandomSetSeed(random, (unsigned long)rank));
  PetscCall(PetscRandomSeed(random));

  // Get a representation of the particle coordinates.
  PetscCall(DMSwarmGetField(
            *swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(*swarm, &np));

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
            *swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Destroy the random-number generator.
  PetscCall(PetscRandomDestroy(&random));

  // Update the swarm.
  PetscCall(DMSwarmMigrate(*swarm, PETSC_TRUE));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeParticles(DM *swarm, Context *ctx)
{
  PetscInt    np;
  PetscScalar *coords;
  RealVector  *pos, *vel;
  PetscInt    ip;

  PetscFunctionBeginUser;

  // Initialize coordinates in the particle DM.
  PetscCall(InitializeSwarmCoordinates(swarm, &ctx));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(*swarm, &np));

  // Get an array representation of the swarm coordinates.
  PetscCall(DMSwarmGetField(
            *swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Get an array representation of the particle positions.
  PetscCall(DMSwarmGetField(
            *swarm,
            "position", NULL, NULL,
            (void **)&pos));

  // Get an array representation of the particle velocities.
  PetscCall(DMSwarmGetField(
            *swarm,
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
            *swarm,
            "position", NULL, NULL,
            (void **)&pos));

  // Restore the particle-velocities array.
  PetscCall(DMSwarmRestoreField(
            *swarm,
            "velocity", NULL, NULL,
            (void **)&vel));

  // Restore the swarm-coordinates array.
  PetscCall(DMSwarmRestoreField(
            *swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Display information about the particle DM.
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CollectParticles(DM *swarm, Context *ctx, Vec gridvec)
{
  DM          grid;
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

  // Get the grid DM from the swarm DM.
  PetscCall(DMSwarmGetCellDM(*swarm, &grid));

  // Make sure the local grid vector has zeroes everywhere.
  PetscCall(VecZeroEntries(gridvec));

  // Get a 4-D array corresponding to the local grid quantities.
  PetscCall(DMDAVecGetArray(grid, gridvec, &array));

  // Get an array representation of the particle positions.
  PetscCall(DMSwarmGetField(
            *swarm,
            "position", NULL, NULL,
            (void **)&pos));

  // Get an array representation of the particle velocities.
  PetscCall(DMSwarmGetField(
            *swarm,
            "velocity", NULL, NULL,
            (void **)&vel));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(*swarm, &np));

  // Compute grid grid spacing.
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

  // Restore the particle-positions array.
  PetscCall(DMSwarmRestoreField(
            *swarm,
            "position", NULL, NULL,
            (void **)&pos));

  // Restore the particle-velocities array.
  PetscCall(DMSwarmRestoreField(
            *swarm,
            "velocity", NULL, NULL,
            (void **)&vel));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
WriteHDF5(DM grid, Vec full, PetscViewer viewer)
{
  PetscInt nFields;
  char **fieldNames;
  IS *is;
  DM *fieldArray;
  PetscInt field;
  Vec fieldVec;

  PetscFunctionBeginUser;

  PetscCall(DMCreateFieldDecomposition(
            grid, &nFields, &fieldNames, &is, &fieldArray));
  for (field=0; field<nFields; field++) {
    PetscCall(DMGetGlobalVector(fieldArray[field], &fieldVec));
    PetscCall(VecStrideGather(full, field, fieldVec, INSERT_VALUES));
    PetscCall(PetscObjectSetName((PetscObject)fieldVec, fieldNames[field]));
    PetscCall(VecView(fieldVec, viewer));
    PetscCall(DMRestoreGlobalVector(fieldArray[field], &fieldVec));
  }

  // TODO: Free memory (see documentation for DMCreateFieldDecomposition)

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
ComputeRHS(KSP ksp, Vec rhs, void *_ctx)
{
  PetscFunctionBeginUser;

  PetscCall(VecSetRandom(rhs, NULL));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ComputeLHS(KSP ksp, Mat J, Mat A, void *_ctx)
{
  // the problem context
  Context      *ctx=(Context *)_ctx;
  // the grid DM of this solver context
  DM           grid;
  // components of magnetization vector
  PetscScalar  Kx, Ky, Kz;
  // components of magnetization tensor
  PetscScalar  rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz;
  // number of cells in each dimension of the global grid
  PetscInt     Nx, Ny, Nz;
  // grid-cell spacing in each dimension
  PetscReal    dx, dy, dz;
  // matrix determinant
  PetscReal    detA;
  // indices of the lower left corner of the local grid
  PetscInt     i0, j0, k0;
  // number of non-ghost cells in each dimension of the local grid
  PetscInt     ni, nj, nk;
  // geometric scale factors
  PetscScalar  sxx, syx, szx, sxy, syy, szy, sxz, syz, szz;
  // local grid vector
  Vec          gridvec;
  // array representation of grid quantities
  GridNode  ***array;
  // grid indices
  PetscInt     i, j, k;
  // the density value at the current and neighboring grid points
  PetscScalar  nijk, npjk, nmjk, nipk, nimk, nijp, nijm;
  // diagonal coefficient
  PetscScalar  vijk;
  // star-stencil coefficients
  PetscScalar  vpjk, vmjk, vipk, vimk, vijp, vijm;
  // x-y corners
  PetscScalar  vmmk, vpmk, vmpk, vppk;
  // x-z corners
  PetscScalar  vpjp, vpjm, vmjp, vmjm;
  // y-z corners
  PetscScalar  vipp, vipm, vimp, vimm;
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

  // Get the grid DM associated with the Krylov solver.
  PetscCall(KSPGetDM(ksp, &grid));

  // Compute grid-cell spacing. Note that the number of grid cells may depend on
  // the Krylov context (e.g., when using multigrid methods), so `N{x,y,z} !=
  // ctx->grid.N{x,y,z}` in general.
  PetscCall(DMDAGetInfo(
            grid, NULL,
            &Nx, &Ny, &Nz,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  dx = 1.0 / (PetscReal)Nx;
  dy = 1.0 / (PetscReal)Ny;
  dz = 1.0 / (PetscReal)Nz;

  /*
  Compute geometric scale factors for stencil values. Note that there is some
  redundancy for the sake of organization.

  Diagonal factors have the form
  - `sii = 2*di*dj*dk / (2*di^2*|A|) = dj*dk / (di*|A|)`

  Off-diagonal factors have the form
  - `sij = 2*di*dj*dk / (8*di*dj*|A|) = dk / (4*|A|)`
  */
  sxx = dy*dz/dx / detA; // 2*dxdydz / (|A|*2*dx^2)
  syx = 0.25*dz  / detA; // 2*dxdydz / (|A|*8*dydx)
  szx = 0.25*dy  / detA; // 2*dxdydz / (|A|*8*dzdx)

  sxy = 0.25*dz  / detA; // 2*dxdydz / (|A|*8*dxdy)
  syy = dx*dz/dy / detA; // 2*dxdydz / (|A|*2*dy^2)
  szy = 0.25*dx  / detA; // 2*dxdydz / (|A|*8*dzdy)

  sxz = 0.25*dy  / detA; // 2*dxdydz / (|A|*8*dxdz)
  syz = 0.25*dx  / detA; // 2*dxdydz / (|A|*8*dydz)
  szz = dx*dy/dz / detA; // 2*dxdydz / (|A|*2*dz^2)

  // Extract the density array.
  PetscCall(DMGetLocalVector(grid, &gridvec));
  PetscCall(DMDAVecGetArray(grid, gridvec, &array));

  // Get this processor's indices.
  PetscCall(DMDAGetCorners(grid, &i0, &j0, &k0, &ni, &nj, &nk));

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
        /* star-stencil coefficients

          The explicit definitions of these coefficients are

          * (i+1, j, k): vpjk =  sxx*rxx*(npjk + nijk) + syx*ryx*(nipk - nimk) + szx*rzx*(nijp - nijm);
          * (i-1, j, k): vmjk =  sxx*rxx*(nijk + nmjk) - syx*ryx*(nipk - nimk) - szx*rzx*(nijp - nijm);
          * (i, j+1, k): vipk =  sxy*rxy*(npjk - nmjk) + syy*ryy*(nipk + nijk) + szy*rzy*(nijp - nijm);
          * (i, j-1, k): vimk = -sxy*rxy*(npjk - nmjk) + syy*ryy*(nijk + nimk) - szy*rzy*(nijp - nijm);
          * (i, j, k+1): vijp =  sxz*rxz*(npjk - nmjk) + syz*ryz*(nipk - nimk) + szz*rzz*(nijp + nijk);
          * (i, j, k-1): vijm = -sxz*rxz*(npjk - nmjk) - syz*ryz*(nipk - nimk) + szz*rzz*(nijk + nijm);

          However, we can equivalently define them with respect to the
          previously-defined corner coefficients, as shown below.
        */
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

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  MPIContext  mpi;
  Context     ctx;
  DM          grid, swarm, solve;
  KSP         ksp;
  Vec         gvec, lvec;
  PetscViewer viewer;
  Vec         phi;

  PetscFunctionBeginUser;

  // Initialize PETSc and MPI.
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n********** START **********\n\n"));

  // Assign parameter values from user arguments or defaults.
  PetscCall(ProcessOptions(&ctx));

  // Store MPI information in the application context.
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &mpi.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi.size));
  ctx.mpi = mpi;

  // Set up discrete grid.
  PetscCall(InitializeGridDM(&grid, &ctx));

  // Set up particle swarm.
  PetscCall(InitializeSwarmDM(&swarm, &grid, &ctx));

  // Set initial particle positions and velocities.
  PetscCall(InitializeParticles(&swarm, &ctx));

  // Compute initial density and flux.
  PetscCall(DMCreateGlobalVector(grid, &gvec));
  PetscCall(VecZeroEntries(gvec));
  PetscCall(DMGetLocalVector(grid, &lvec));
  PetscCall(DMGlobalToLocalBegin(grid, gvec, INSERT_VALUES, lvec));
  PetscCall(DMGlobalToLocalEnd(grid, gvec, INSERT_VALUES, lvec));
  PetscCall(CollectParticles(&swarm, &ctx, lvec));
  PetscCall(DMLocalToGlobalBegin(grid, lvec, ADD_VALUES, gvec));
  PetscCall(DMLocalToGlobalEnd(grid, lvec, ADD_VALUES, gvec));
  PetscCall(DMRestoreLocalVector(grid, &lvec));

  // [DEV] View the global grid vector.
  PetscCall(PetscViewerHDF5Open(
            PETSC_COMM_WORLD, "grid.hdf", FILE_MODE_WRITE, &viewer));
  PetscCall(WriteHDF5(grid, gvec, viewer));

  // Set up the linear-solver context.
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(InitializePotentialDM(grid, &solve));
  PetscCall(KSPSetDM(ksp, solve));
  PetscCall(KSPSetFromOptions(ksp));

  // Compute initial electric field.
  PetscCall(KSPSetComputeInitialGuess(ksp, ComputeInitialPhi, &ctx));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, &ctx));
  PetscCall(KSPSetComputeOperators(ksp, ComputeLHS, &ctx));

  // Solve the system.
  PetscCall(KSPSolve(ksp, NULL, NULL));
  PetscCall(KSPGetSolution(ksp, &phi));

  // Output initial conditions.

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
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&gvec));
  PetscCall(DMDestroy(&grid));
  PetscCall(DMDestroy(&swarm));

  // Finalize PETSc and MPI.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n*********** END ***********\n"));
  PetscCall(PetscFinalize());

  return 0;
}