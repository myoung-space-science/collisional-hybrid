/* 3D Hybrid PIC */
static char help[] = "A 3D hybrid particle-in-cell (PIC) simulation.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>


#define NDIM 3
#define Q 1.6022e-19  // fundamental charge in C
#define MP 1.6726e-27 // proton mass in kg

// Temporary declaration of number of particles per cell, per dimension.
#define NPPCELL 1

typedef struct {
  PetscInt nx;  // number of cells in x dimension
  PetscInt ny;  // number of cells in y dimension
  PetscInt nz;  // number of cells in z dimension
  PetscReal Lx; // length of x dimension
  PetscReal Ly; // length of y dimension
  PetscReal Lz; // length of z dimension
} UserGrid;

typedef struct {
  PetscInt np;   // number of particles
  PetscReal q;   // the charge of each particle / fundamental charge
  PetscReal m;   // the mass of each particle / proton mass
  PetscReal nu;  // the neutral-collision frequency of each particle
  PetscReal vx;  // initial x velocity
  PetscReal vy;  // initial y velocity
  PetscReal vz;  // initial z velocity
} UserPIC;

typedef struct {
  UserGrid grid; // grid information
  UserPIC pic;   // particle information
} UserContext;

typedef struct {
  PetscReal q;   // charge
  PetscReal m;   // mass
  PetscReal nu;  // frequency of collisions with neutral particles
  PetscReal x;   // x position
  PetscReal y;   // y position
  PetscReal z;   // z position
  PetscReal vx;  // x velocity
  PetscReal vy;  // y velocity
  PetscReal vz;  // z velocity
} Species;

typedef struct {
  PetscMPIInt rank;
  PetscMPIInt size;
} MPIContext;

typedef struct {
  UserContext user;
  MPIContext  mpi;
} Context;

typedef struct {
  PetscScalar n;          // scalar density
  PetscScalar flux[NDIM]; // vector flux
  PetscScalar phi;        // scalar electrostatic potential
} GridNode;

static PetscErrorCode
ProcessOptions(UserContext *options)
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

  // Declare default parameter values. The default value of nx, ny, and nz is
  // set such that it will signal to the grid-setup routine whether the user
  // provided a non-negative value for each. That will allow it to properly
  // handle cases when the user provides equivalent values via PETSc's
  // -da_grid_{x,y,z} flags.
  options->grid.nx = -1;
  options->grid.ny = -1;
  options->grid.nz = -1;
  options->grid.Lx = 1.0;
  options->grid.Ly = 1.0;
  options->grid.Lz = 1.0;
  options->pic.q = 1.0;
  options->pic.m = 1.0;
  options->pic.nu = 1.0;
  options->pic.vx = 0.0;
  options->pic.vy = 0.0;
  options->pic.vz = 0.0;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &intArg, &found));
  if (found) {
    if (intArg < 0) {
      PetscCall(PetscPrintf(
                PETSC_COMM_WORLD,
                "Warning: Ignoring negative value for nx: %d\n", intArg));
    } else {
      options->grid.nx = intArg;
    }
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ny", &intArg, &found));
  if (found) {
    if (intArg < 0) {
      PetscCall(PetscPrintf(
                PETSC_COMM_WORLD,
                "Warning: Ignoring negative value for ny: %d\n", intArg));
    } else {
      options->grid.ny = intArg;
    }
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nz", &intArg, &found));
  if (found) {
    if (intArg < 0) {
      PetscCall(PetscPrintf(
                PETSC_COMM_WORLD,
                "Warning: Ignoring negative value for nz: %d\n", intArg));
    } else {
      options->grid.nz = intArg;
    }
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Lx", &realArg, &found));
  if (found) {
    options->grid.Lx = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Ly", &realArg, &found));
  if (found) {
    options->grid.Ly = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Lz", &realArg, &found));
  if (found) {
    options->grid.Lz = realArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-np", &intArg, &found));
  if (found) {
    options->pic.np = intArg;
  } else {
    options->pic.np = options->grid.nx * options->grid.ny * options->grid.nz;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-q", &realArg, &found));
  if (found) {
    options->pic.q = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-m", &realArg, &found));
  if (found) {
    options->pic.m = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-nu", &realArg, &found));
  if (found) {
    options->pic.nu = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vx", &realArg, &found));
  if (found) {
    options->pic.vx = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vy", &realArg, &found));
  if (found) {
    options->pic.vy = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vz", &realArg, &found));
  if (found) {
    options->pic.vz = realArg;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CreateGridDM(DM *grid, Context *ctx)
{
  PetscInt       nx=(ctx->user.grid.nx > 0 ? ctx->user.grid.nx : 7);
  PetscInt       ny=(ctx->user.grid.ny > 0 ? ctx->user.grid.ny : 7);
  PetscInt       nz=(ctx->user.grid.nz > 0 ? ctx->user.grid.nz : 7);
  DMBoundaryType xBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType yBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType zBC=DM_BOUNDARY_PERIODIC;
  PetscInt       dof=5;
  PetscInt       width=1;

  PetscFunctionBeginUser;

  PetscCall(DMDACreate3d(
            PETSC_COMM_WORLD,
            xBC, yBC, zBC,
            DMDA_STENCIL_BOX,
            nx, ny, nz,
            PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
            dof, width,
            NULL, NULL, NULL,
            grid));
  PetscCall(DMDASetElementType(*grid, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(*grid));
  PetscCall(DMSetUp(*grid));
  // Coordinate values of nx, ny, and nz passed via -n{x,y,z} or
  // -da_grid_{x,y,z}. Note that this gives precedence to the latter.
  PetscCall(DMDAGetInfo(
            *grid, NULL,
            &nx, &ny, &nz,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  if (ctx->user.grid.nx == -1) {
    ctx->user.grid.nx = nx;
  }
  if (ctx->user.grid.ny == -1) {
    ctx->user.grid.ny = ny;
  }
  if (ctx->user.grid.nz == -1) {
    ctx->user.grid.nz = nz;
  }
  PetscCall(DMDASetUniformCoordinates(
            *grid,
            0.0, ctx->user.grid.Lx,
            0.0, ctx->user.grid.Ly,
            0.0, ctx->user.grid.Lz));
  PetscCall(DMDASetFieldName(*grid, 0, "density"));
  PetscCall(DMDASetFieldName(*grid, 1, "x flux"));
  PetscCall(DMDASetFieldName(*grid, 2, "y flux"));
  PetscCall(DMDASetFieldName(*grid, 3, "z flux"));
  PetscCall(DMDASetFieldName(*grid, 4, "potential"));
  PetscCall(DMSetApplicationContext(*grid, &ctx->user));
  PetscCall(DMView(*grid, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CreateSwarmDM(DM *swarm, DM *grid, Context *ctx)
{
  PetscInt dim;
  PetscInt bufsize=0;
  PetscInt np;

  PetscFunctionBeginUser;

  PetscCall(DMCreate(PETSC_COMM_WORLD, swarm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*swarm), "pic_"));
  PetscCall(DMSetType(*swarm, DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject)*swarm, "Ions"));
  PetscCall(DMGetDimension(*grid, &dim));
  PetscCall(DMSetDimension(*swarm, dim));
  PetscCall(DMSwarmSetType(*swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*swarm, *grid));
  PetscCall(DMSwarmInitializeFieldRegister(*swarm));
  PetscCall(DMSwarmRegisterUserStructField(
            *swarm, "Species", sizeof(Species)));
  PetscCall(DMSwarmFinalizeFieldRegister(*swarm));
  // Set the per-processor swarm size and buffer length for efficient resizing.
  np = ctx->user.pic.np / ctx->mpi.size;
  PetscCall(DMSwarmSetLocalSizes(*swarm, np, bufsize));
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
  Species     *params;
  PetscInt    ip;

  PetscFunctionBeginUser;

  // Initialize coordinates in the particle DM.
  PetscCall(InitializeSwarmCoordinates(swarm, &ctx->user));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(*swarm, &np));

  // Get a representation of the particle coordinates.
  PetscCall(DMSwarmGetField(
            *swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Get a representation of the particle parameters.
  PetscCall(DMSwarmGetField(
            *swarm,
            "Species", NULL, NULL,
            (void **)&params));

  // Loop over particles and assign parameter values.
  for (ip=0; ip<np; ip++) {
    params[ip].q  = Q * ctx->user.pic.q;
    params[ip].m  = MP * ctx->user.pic.m;
    params[ip].nu = ctx->user.pic.nu;
    params[ip].x  = coords[ip*NDIM + 0];
    params[ip].y  = coords[ip*NDIM + 1];
    params[ip].z  = coords[ip*NDIM + 2];
    params[ip].vx = ctx->user.pic.vx;
    params[ip].vy = ctx->user.pic.vy;
    params[ip].vz = ctx->user.pic.vz;
  }

  // Restore the parameters array.
  PetscCall(DMSwarmRestoreField(
            *swarm,
            "Species", NULL, NULL,
            (void **)&params));

  // Restore the coordinates array.
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
  GridNode   ***array;
  PetscInt    dim;
  PetscInt    i0, j0, k0;
  PetscReal   x, y, z, dx, dy, dz;
  Species     *params, current;
  PetscInt    ip, np;
  PetscInt    ixl, ixh, iyl, iyh, izl, izh;
  PetscReal   wxl, wxh, wyl, wyh, wzl, wzh;
  PetscReal   hhh, lhh, hlh, llh, hhl, lhl, hll, lll;
  PetscReal   v[NDIM];

  PetscFunctionBeginUser;

  // Get the grid DM from the swarm DM.
  PetscCall(DMSwarmGetCellDM(*swarm, &grid));

  // Make sure the local grid vector has zeroes everywhere.
  PetscCall(VecZeroEntries(gridvec));

  // Get a 4-D array corresponding to the local grid quantities.
  PetscCall(DMDAVecGetArray(grid, gridvec, &array));

  // Get a representation of the particle parameters.
  PetscCall(DMSwarmGetField(
            *swarm,
            "Species", NULL, NULL,
            (void **)&params));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(*swarm, &np));

  // Compute grid grid spacing.
  PetscCall(DMDAGetInfo(
            grid, NULL,
            &i0, &j0, &k0,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  dx = ctx->user.grid.Lx / (PetscReal)i0;
  dy = ctx->user.grid.Ly / (PetscReal)j0;
  dz = ctx->user.grid.Lz / (PetscReal)k0;

  // Loop over particles.
  for (ip=0; ip<np; ip++) {
    // Get the current particle's parameters.
    current = params[ip];
    x = current.x / dx;
    y = current.y / dy;
    z = current.z / dz;
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
    v[0] = current.vx;
    v[1] = current.vy;
    v[2] = current.vz;
    for (dim=0; dim<NDIM; dim++) {
      array[izh][iyh][ixh].flux[dim] += v[dim]*hhh;
      array[izl][iyh][ixh].flux[dim] += v[dim]*lhh;
      array[izh][iyl][ixh].flux[dim] += v[dim]*hlh;
      array[izl][iyl][ixh].flux[dim] += v[dim]*llh;
      array[izh][iyh][ixl].flux[dim] += v[dim]*hhl;
      array[izl][iyh][ixl].flux[dim] += v[dim]*lhl;
      array[izh][iyl][ixl].flux[dim] += v[dim]*hll;
      array[izl][iyl][ixl].flux[dim] += v[dim]*lll;
    }
  }

  // Restore the local grid array.
  PetscCall(DMDAVecRestoreArray(grid, gridvec, &array));

  // Restore the parameters array.
  PetscCall(DMSwarmRestoreField(
            *swarm,
            "Species", NULL, NULL,
            (void **)&params));

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

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  UserContext user;
  MPIContext  mpi;
  Context     ctx;
  DM          grid, swarm;
  KSP         ksp;
  PetscMPIInt rank, size;
  Vec         gvec, lvec;
  PetscViewer      viewer;

  PetscFunctionBeginUser;

  // Initialize PETSc and MPI.
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n********** START **********\n\n"));

  // Store MPI information in the application context.
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &mpi.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi.size));
  ctx.mpi = mpi;

  // Assign parameter values from user arguments or defaults.
  PetscCall(ProcessOptions(&user));
  ctx.user = user;

  // Set up discrete grid.
  PetscCall(CreateGridDM(&grid, &ctx));

  // Set up particle swarm.
  PetscCall(CreateSwarmDM(&swarm, &grid, &ctx));

  // Set up the linear-solver context.
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, grid));

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

  // Compute initial electric field.

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
  PetscCall(DMDestroy(&grid));
  PetscCall(DMDestroy(&swarm));
  PetscCall(VecDestroy(&gvec));
  PetscCall(PetscViewerDestroy(&viewer));

  // Finalize PETSc and MPI.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n*********** END ***********\n"));
  PetscCall(PetscFinalize());

  return 0;
}