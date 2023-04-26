/* 3D Hybrid PIC */
static char help[] = "A 3D hybrid particle-in-cell (PIC) simulation.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>


#define NDIM 3

typedef struct {
  PetscInt nx;  // number of cells in x dimension
  PetscInt ny;  // number of cells in y dimension
  PetscInt nz;  // number of cells in z dimension
  PetscReal Lx; // length of x dimension
  PetscReal Ly; // length of y dimension
  PetscReal Lz; // length of z dimension
} UserMesh;

typedef struct {
  PetscInt np; // number of particles
} UserPIC;

typedef struct {
  UserMesh grid; // grid information
  UserPIC pic;   // particle information
} UserContext;

typedef struct {
  PetscReal q;   // charge
  PetscReal m;   // mass
  PetscReal nue; // frequency of collisions with electrons
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
  PetscInt  np;

  options->grid.nx = 7;
  options->grid.ny = 7;
  options->grid.nz = 7;
  np = options->grid.nx * options->grid.ny * options->grid.nz;
  options->pic.np = np;
  options->grid.Lx = 1.0;
  options->grid.Ly = 1.0;
  options->grid.Lz = 1.0;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &intArg, &found));
  if (found) {
    options->grid.nx = intArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ny", &intArg, &found));
  if (found) {
    options->grid.ny = intArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nz", &intArg, &found));
  if (found) {
    options->grid.nz = intArg;
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
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CreateMeshDM(DM *mesh, UserContext *user)
{
  PetscInt       nx=user->grid.nx;
  PetscInt       ny=user->grid.ny;
  PetscInt       nz=user->grid.nz;
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
InitializeParticles(DM *mesh, DM *swarm, UserContext *user, PetscInt n0pc)
{
  PetscInt    np;
  PetscScalar *coords;
  PetscReal   dx, dy, dz, x, y, z;
  PetscInt    i, j, k;
  int         size;

  PetscFunctionBeginUser;

  PetscCall(DMSwarmInsertPointsUsingCellDM(
            *swarm, DMSWARMPIC_LAYOUT_REGULAR, n0pc));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  np = (PetscInt)PetscCbrtReal((PetscReal)(user->pic.np / size));
  dx = user->grid.Lx / PetscMax(1, np-1);
  dy = user->grid.Ly / PetscMax(1, np-1);
  dz = user->grid.Lz / PetscMax(1, np-1);
  PetscCall(DMSwarmGetField(
            *swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));
  x = 0.0;
  for (i=0; i<np; ++i, x += dx) {
    y = 0.0;
    for (j=0; j<np; ++j, y += dy) {
      z = 0.0;
      for (k=0; k<np; ++k, z += dz) {
        const PetscInt p = (k*np + j)*np + i;
        coords[p*NDIM + 0] = x;
        coords[p*NDIM + 1] = y;
        coords[p*NDIM + 2] = z;
      }
    }
  }
  PetscCall(DMSwarmRestoreField(
            *swarm,
            DMSwarmPICField_coor, NULL, NULL,
            (void **)&coords));
  PetscCall(DMSwarmMigrate(*swarm, PETSC_TRUE));

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
  PetscCall(InitializeParticles(&mesh, &swarm, &user, 3));

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