/* 3D Hybrid PIC */
static char help[] = "A 3D hybrid particle-in-cell (PIC) simulation.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>


typedef struct {
  PetscInt nx, ny, nz;
  PetscReal Lx, Ly, Lz;
} Grid;

typedef struct {
  PetscInt n;
} Particles;

typedef struct {
  Grid grid;
  Particles particles;
} UserContext;


static PetscErrorCode ProcessOptions(UserContext *options)
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

  options->grid.nx = 7;
  options->grid.ny = 7;
  options->grid.nz = 7;
  options->particles.n = 7;
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
    options->particles.n = intArg;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode CreateMeshDM(DM *mesh, UserContext *user)
{
  PetscInt       nx=11;
  PetscInt       ny=11;
  PetscInt       nz=11;
  DMBoundaryType xBC=DM_BOUNDARY_GHOSTED;
  DMBoundaryType yBC=DM_BOUNDARY_GHOSTED;
  DMBoundaryType zBC=DM_BOUNDARY_GHOSTED;
  PetscInt       dof=2;
  PetscInt       width=1;

  PetscFunctionBeginUser;

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, xBC, yBC, zBC, DMDA_STENCIL_BOX,
                         nx, ny, nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                         dof, width, NULL, NULL, NULL, mesh));
  PetscCall(DMDASetElementType(*mesh, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(*mesh));
  PetscCall(DMSetUp(*mesh));
  PetscCall(DMDASetUniformCoordinates(*mesh,
                                      0.0, user->grid.Lx,
                                      0.0, user->grid.Ly,
                                      0.0, user->grid.Lz));
  PetscCall(DMSetApplicationContext(*mesh, user));
  PetscCall(DMView(*mesh, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode CreateSwarmDM(DM *swarm, DM *mesh)
{
  PetscInt dim;

  PetscFunctionBeginUser;

  PetscCall(DMCreate(PETSC_COMM_WORLD, swarm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*swarm), "pic_"));
  PetscCall(DMSetType(*swarm, DMSWARM));
  PetscCall(DMGetDimension(*mesh, &dim));
  PetscCall(DMSetDimension(*swarm, dim));
  PetscCall(DMSwarmSetType(*swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*swarm, *mesh));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*swarm,
                                              "potential", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*swarm,
                                              "density", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*swarm));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

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
  PetscCall(CreateSwarmDM(&swarm, &mesh));

  // Set up the linear-solver context.
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, mesh));

  // Set initial particle positions and velocities.

  // Compute initial density and electric field.

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

  // Finalize PETSc and MPI.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n*********** END ***********\n"));
  PetscCall(PetscFinalize());

  return 0;
}