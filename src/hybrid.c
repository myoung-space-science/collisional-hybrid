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
#define NPPCELL 3

typedef struct {
  PetscInt nx;  // number of cells in x dimension
  PetscInt ny;  // number of cells in y dimension
  PetscInt nz;  // number of cells in z dimension
  PetscReal Lx; // length of x dimension
  PetscReal Ly; // length of y dimension
  PetscReal Lz; // length of z dimension
} UserMesh;

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
  UserMesh grid; // grid information
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
CreateMeshDM(DM *mesh, UserContext *user)
{
  PetscInt       nx=(user->grid.nx > 0 ? user->grid.nx : 7);
  PetscInt       ny=(user->grid.ny > 0 ? user->grid.ny : 7);
  PetscInt       nz=(user->grid.nz > 0 ? user->grid.nz : 7);
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
            mesh));
  PetscCall(DMDASetElementType(*mesh, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(*mesh));
  PetscCall(DMSetUp(*mesh));
  // Coordinate values of nx, ny, and nz passed via -n{x,y,z} or
  // -da_grid_{x,y,z}. Note that this gives precedence to the latter.
  PetscCall(DMDAGetInfo(
            *mesh, NULL,
            &nx, &ny, &nz,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  if (user->grid.nx == -1) {
    user->grid.nx = nx;
  }
  if (user->grid.ny == -1) {
    user->grid.ny = ny;
  }
  if (user->grid.nz == -1) {
    user->grid.nz = nz;
  }
  PetscCall(DMDASetUniformCoordinates(
            *mesh,
            0.0, user->grid.Lx,
            0.0, user->grid.Ly,
            0.0, user->grid.Lz));
  PetscCall(DMDASetFieldName(*mesh, 0, "density"));
  PetscCall(DMDASetFieldName(*mesh, 1, "x flux"));
  PetscCall(DMDASetFieldName(*mesh, 1, "y flux"));
  PetscCall(DMDASetFieldName(*mesh, 1, "z flux"));
  PetscCall(DMDASetFieldName(*mesh, 4, "potential"));
  PetscCall(DMSetApplicationContext(*mesh, user));
  PetscCall(DMView(*mesh, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CreateSwarmDM(DM *swarm, DM *mesh, UserContext *user)
{
  PetscInt dim;
  PetscInt bufsize=0;
  PetscInt np;
  int      size;

  PetscFunctionBeginUser;

  PetscCall(DMCreate(PETSC_COMM_WORLD, swarm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*swarm), "pic_"));
  PetscCall(DMSetType(*swarm, DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject)*swarm, "Ions"));
  PetscCall(DMGetDimension(*mesh, &dim));
  PetscCall(DMSetDimension(*swarm, dim));
  PetscCall(DMSwarmSetType(*swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*swarm, *mesh));
  PetscCall(DMSwarmInitializeFieldRegister(*swarm));
  PetscCall(DMSwarmRegisterUserStructField(
            *swarm, "Species", sizeof(Species)));
  PetscCall(DMSwarmFinalizeFieldRegister(*swarm));
  // Set the per-processor swarm size and buffer length for efficient resizing.
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  np = user->pic.np / size;
  PetscCall(DMSwarmSetLocalSizes(*swarm, np, bufsize));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeSwarmCoordinates(DM *swarm, UserContext *user)
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
InitializeParticles(DM *swarm, UserContext *user)
{
  PetscInt    np;
  PetscScalar *coords;
  Species     *params;
  PetscInt    ip;

  PetscFunctionBeginUser;

  // Initialize coordinates in the particle DM.
  PetscCall(InitializeSwarmCoordinates(swarm, user));

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
    params[ip].q  = Q * user->pic.q;
    params[ip].m  = MP * user->pic.m;
    params[ip].nu = user->pic.nu;
    params[ip].x  = coords[ip*NDIM + 0];
    params[ip].y  = coords[ip*NDIM + 1];
    params[ip].z  = coords[ip*NDIM + 2];
    params[ip].vx = user->pic.vx;
    params[ip].vy = user->pic.vy;
    params[ip].vz = user->pic.vz;
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
CollectParticles(DM *mesh, DM *swarm, UserContext *user)
{
  PetscScalar *coords;
  PetscInt    i, i0, ni, j, j0, nj, k, k0, nk;
  PetscInt    np;

  PetscFunctionBeginUser;

  // Get density Vec from mesh.

  // Get array representation of density Vec.

  // Same for flux components.

  // Get the particle coordinates.
  PetscCall(DMSwarmGetField(
            *swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(*swarm, &np));

  // Loop over particle positions.

    // Assign n[k][j][i] = 3D linear weight.

    // Assign Gx[k][j][i] = 3D linear weight * vx.

    // Assign Gy[k][j][i] = 3D linear weight * vy.

    // Assign Gz[k][j][i] = 3D linear weight * vz.

  // Restore the particle coordinates.
  PetscCall(DMSwarmRestoreField(
            *swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));

  // Restore Vec array representations.

  // Restore mesh Vec objects.

  // Assemble Vec objects?

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  UserContext user;
  DM          mesh, swarm;
  KSP         ksp;

  PetscFunctionBeginUser;

  // Initialize PETSc and MPI.
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n********** START **********\n\n"));

  // Assign parameter values from user arguments or defaults.
  PetscCall(ProcessOptions(&user));

  // Set up discrete mesh.
  PetscCall(CreateMeshDM(&mesh, &user));

  // Set up particle swarm.
  PetscCall(CreateSwarmDM(&swarm, &mesh, &user));

  // Set up the linear-solver context.
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, mesh));

  // Set initial particle positions and velocities.
  PetscCall(InitializeParticles(&swarm, &user));

  // Compute initial density and electric field.
  PetscCall(CollectParticles(&mesh, &swarm, &user));

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
  PetscCall(DMDestroy(&mesh));
  PetscCall(DMDestroy(&swarm));

  // Finalize PETSc and MPI.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n*********** END ***********\n"));
  PetscCall(PetscFinalize());

  return 0;
}