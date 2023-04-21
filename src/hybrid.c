/* 3D Hybrid PIC */
static char help[] = "A 3D hybrid particle-in-cell (PIC) simulation.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>


int main(int argc, char **args)
{
  PetscFunctionBeginUser;

  // Initialize PETSc and MPI.
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  // Assign parameter values from user arguments or defaults.

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