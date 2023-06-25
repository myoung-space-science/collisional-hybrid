/* 3-D Electrostatic Potential Solver */
static char help[] = "A tool for solving the 3D quasineutral electrostatic-potential equation.";

#include <time.h>
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include <slepceps.h>
#include "hybrid.h"
#include "parameters.h"
#include "setup.h"
#include "particles.h"
#include "potential.h"
#include "fileio.h"


int main(int argc, char **args)
{
  MPIContext  mpi;
  time_t      startTime, endTime;
  Context     ctx;
  DM          pdm;
  KSP         ksp;

  PetscFunctionBeginUser;

  /* Initialize PETSc and MPI. */
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &mpi.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi.size));

  /* Log start time. */
  time(&startTime);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n**************** START *****************\n\n"));

  /* Initialize SLEPc. */
  PetscCall(SlepcInitialize(&argc, &args, (char *)0, help));

  /* Assign parameter values from user arguments or defaults. */
  PetscCall(ProcessOptions(&ctx));

  /* Store MPI information in the application context. */
  ctx.mpi = mpi;

  /* Set up the discrete grid of Vlasov quantities. */
  PetscCall(SetUpVlasovDM(&ctx));

  /* Echo the initial state. */
  PetscCall(EchoOptions(ctx));

  /* Read density and fluxes from disk. */
  PetscCall(LoadVlasovQuantities(&ctx));

  /* Set up the discrete grid for the electrostatic potential. */
  PetscCall(SetUpPotentialDM(&pdm, &ctx));

  /* Set up the Krylov-solver context for the electrostatic potential. */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, pdm));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetComputeInitialGuess(ksp, ComputeInitialPhi, &ctx));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, &ctx));
  PetscCall(KSPSetComputeOperators(ksp, ComputeLHS, &ctx));

  /* Compute the electrostatic potential. */
  PetscCall(ComputePotential(ksp, &ctx));

  /* Output arrays. */
  PetscCall(OutputHDF5("", &ctx));

  /* Compute LHS eigenvalues. */
  {
    PetscBool found, true;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "--lhs-eigenvalues", &true, &found));
    if (found && true) {
      PetscCall(ComputeLHSEigenvalues(ksp));
    }
  }

  /* Free memory. */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&ctx.vlasov));
  PetscCall(DMDestroy(&pdm));

  /* Log end time. */
  time(&endTime);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n----------------------------------------\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Start time: %s", asctime(localtime(&startTime))));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "End time:   %s", asctime(localtime(&endTime))));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Elapsed time: %f s\n", (float)(endTime-startTime)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "----------------------------------------\n"));

  /* Finalize SLEPc. */
  PetscCall(SlepcFinalize());

  /* Finalize PETSc and MPI. */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n***************** END ******************\n"));
  PetscCall(PetscFinalize());

  return 0;
}
