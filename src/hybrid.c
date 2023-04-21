/* 3D Hybrid PIC */
static char help[] = "A 3D hybrid particle-in-cell (PIC) simulation.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>


typedef struct {
  PetscInt nx, ny, nz;
} Grid;
typedef struct {
  Grid grid;
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

  PetscCall(PetscOptionsGetInt(NULL, NULL, "--nx", &intArg, &found));
  if (found) {
    options->grid.nx = &intArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "--ny", &intArg, &found));
  if (found) {
    options->grid.ny = &intArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "--nz", &intArg, &found));
  if (found) {
    options->grid.nz = &intArg;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  UserContext user;

  PetscFunctionBeginUser;

  // Initialize PETSc and MPI.
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  // Assign parameter values from user arguments or defaults.
  PetscCall(ProcessOptions(&user));

  // Set initial particle positions and velocities.

  // Compute initial density and electric field.

  // Output initial conditions.

  // Main time-step loop.

    // Compute density and flux from particle positions.

    //  Compute potential from density

    //  Compute electric field from potential

    //  Update velocities: $\frac{d\vec{v}}{dt} = \frac{e\vec{E}}{m_i}$

      //  Apply Boris mover

      //  Apply collisions

    //  Update positions: $\frac{d\vec{r}}{dt} = \vec{v}$

    //  Apply boundary conditions

    //  Output current time step

  // Finalize PETSc and MPI.
  PetscCall(PetscFinalize());

  return 0;
}