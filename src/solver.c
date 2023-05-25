/* 3-D Electrostatic Potential Solver */
static char help[] = "A tool for solving the 3D quasineutral electrostatic-potential equation.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include "hybrid.h"
#include "parameters.h"
#include "setup.h"
#include "particles.h"
#include "potential.h"
#include "fileio.h"


int main(int argc, char **args)
{
  MPIContext  mpi;
  Context     ctx;
  DM          vdm, pdm;
  KSP         ksp;

  PetscFunctionBeginUser;

  /* Initialize PETSc and MPI. */
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &mpi.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi.size));

  /* Assign parameter values from user arguments or defaults. */
  PetscCall(ProcessOptions(&ctx));

  /* Store MPI information in the application context. */
  ctx.mpi = mpi;

  /* Set up the discrete grid of Vlasov quantities. */
  PetscCall(SetUpVlasovDM(&vdm, &ctx));

  /* Create a persistent vector for outputing Vlasov quantities. */
  PetscCall(DMCreateGlobalVector(vdm, &ctx.vlasov));
  PetscCall(VecZeroEntries(ctx.vlasov));

  /* Echo the initial state. */
  if (mpi.rank == 0) {
    PetscCall(EchoOptions(ctx));
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  /* Read density and fluxes from disk. */
  PetscCall(LoadVlasov(vdm, "/home/matthew/sandbox/dmswarm-hybrid/test-data/vlasov.h5", &ctx.vlasov));

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
  PetscCall(OutputHDF5("solver.hdf", &ctx));

  /* Free memory. */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&ctx.vlasov));
  PetscCall(DMDestroy(&pdm));

  /* Finalize PETSc and MPI. */
  PetscCall(PetscFinalize());

  return 0;
}
