/* 3D Hybrid PIC */
static char help[] = "A 3D hybrid particle-in-cell (PIC) simulation.";

#include <time.h>
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include "hybrid.h"
#include "parameters.h"
#include "setup.h"
#include "random.h"
#include "distributions.h"
#include "particles.h"
#include "potential.h"
#include "lhs.h"
#include "rhs.h"
#include "vectors.h"
#include "output.h"


int main(int argc, char **args)
{
  MPIContext  mpi;
  time_t      startTime, endTime;
  Context     ctx;
  DM          vdm, pdm;
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
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n**************** START *****************\n\n"));

  /* Assign parameter values from user arguments or defaults. */
  PetscCall(ProcessOptions(&ctx));

  /* Store MPI information in the application context. */
  ctx.mpi = mpi;

  /* Define a rank-specific random-number seed. */
  ctx.seed = (long)(-(mpi.rank + 1)*12345);

  /* Set up the discrete grid of Vlasov quantities. */
  PetscCall(SetUpVlasovDM(&vdm, &ctx));

  /* Create a persistent vector for outputing Vlasov quantities. */
  PetscCall(DMCreateGlobalVector(vdm, &ctx.vlasov));
  PetscCall(VecZeroEntries(ctx.vlasov));

  /* Set up the particle swarm for ions. */
  PetscCall(SetUpIonsDM(vdm, &ctx));

  /* Set initial particle positions. */
  PetscCall(InitializePositions(&ctx));

  /* Set initial particle velocities. */
  PetscCall(InitializeVelocities(&ctx));

  /* Echo the initial state. */
  if (mpi.rank == 0) {
    PetscCall(EchoOptions(ctx));
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  /* Compute initial density and flux. */
  PetscCall(CollectVlasovQuantities(&ctx));

  /* Set up the discrete grid for the electrostatic potential. */
  PetscCall(SetUpPotentialDM(&pdm, &ctx));

  /* Set up the Krylov-solver context for the electrostatic potential. */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, pdm));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetComputeInitialGuess(ksp, ComputeInitialPhi, &ctx));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, &ctx));
  PetscCall(KSPSetComputeOperators(ksp, ComputeLHS, &ctx));

  /* Compute initial electrostatic potential. */
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
    PetscCall(CollectVlasovQuantities(&ctx));

    /* Compute potential from density. */
    PetscCall(ComputePotential(ksp, &ctx));

    /* Output current time step. */
    sprintf(pathstr, pathfmt, it);
    PetscCall(OutputHDF5(pathstr, &ctx));

  }

  /* Free memory. */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&ctx.vlasov));
  PetscCall(DMDestroy(&vdm));
  PetscCall(DMDestroy(&ctx.ionsDM));
  PetscCall(DMDestroy(&pdm));

  /* Write time information. */
  if (mpi.rank == 0) {
    time(&endTime);
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n----------------------------------------\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Start time: %s", asctime(localtime(&startTime))));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "End time:   %s", asctime(localtime(&endTime))));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "----------------------------------------\n"));

  /* Finalize PETSc and MPI. */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n***************** END ******************\n"));
  PetscCall(PetscFinalize());

  return 0;
}