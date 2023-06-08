/* 3-D Electrostatic Potential Solver */
static char help[] = "A tool for solving the 3D quasineutral electrostatic-potential equation.";

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
  Context     ctx;
  DM          pdm;
  KSP         ksp;
  EPS         eps;
  EPSType     type;
  Mat         A;
  PetscInt    i, its, maxit, nev, nconv;
  PetscReal   tol, error, re, im;
  PetscScalar kr, ki;
  Vec         xr, xi;

  PetscFunctionBeginUser;

  /* Initialize PETSc and MPI. */
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &mpi.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi.size));

  /* Initialize SLEPc. */
  PetscCall(SlepcInitialize(&argc, &args, (char *)0, help));

  /* Assign parameter values from user arguments or defaults. */
  PetscCall(ProcessOptions(&ctx));

  /* Store MPI information in the application context. */
  ctx.mpi = mpi;

  /* Set up the discrete grid of Vlasov quantities. */
  PetscCall(SetUpVlasovDM(&ctx));

  /* Echo the initial state. */
  if (mpi.rank == 0) {
    PetscCall(EchoOptions(ctx));
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  /* Read density and fluxes from disk. */
  PetscCall(LoadVlasov("/home/matthew/sandbox/dmswarm-hybrid/test-data/vlasov.h5", &ctx));

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

  /* Compute the eigenvalues of the operator matrix.
  
  This is essentially a distilled version of ${SLEPC_DIR}/src/eps/tutorials/ex1.c
  */
  PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(MatCreateVecs(A, NULL, &xr));
  PetscCall(MatCreateVecs(A, NULL, &xi));
  PetscCall(EPSSetOperators(eps, A, NULL));
  PetscCall(EPSSetProblemType(eps, EPS_HEP));
  PetscCall(EPSSetFromOptions(eps));
  PetscCall(EPSSolve(eps));
  PetscCall(EPSGetIterationNumber(eps, &its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Number of iterations of the method: %d\n", its));
  PetscCall(EPSGetType(eps, &type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Solution method: %s\n\n", type));
  PetscCall(EPSGetDimensions(eps, &nev, NULL, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Number of requested eigenvalues: %d\n", nev));
  PetscCall(EPSGetTolerances(eps, &tol, &maxit));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Stopping condition: tol=%.4g, maxit=%d\n", (double)tol, maxit));
  PetscCall(EPSGetConverged(eps, &nconv));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Number of converged eigenpairs: %d\n\n", nconv));
  if (nconv > 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
         "           k          ||Ax-kx||/||kx||\n"
         "   ----------------- ------------------\n"));

    for (i=0; i<nconv; i++) {
      PetscCall(EPSGetEigenpair(eps, i, &kr, &ki, xr, xi));
      PetscCall(EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &error));
#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(kr);
      im = PetscImaginaryPart(kr);
#else
      re = kr;
      im = ki;
#endif
      if (im!=0.0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %9f%+9fi %12g\n", (double)re, (double)im, (double)error));
      else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "   %12f       %12g\n", (double)re, (double)error));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }

  /* Free memory. */
  PetscCall(EPSDestroy(&eps));
  PetscCall(VecDestroy(&xr));
  PetscCall(VecDestroy(&xi));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&ctx.vlasov));
  PetscCall(DMDestroy(&pdm));

  /* Finalize SLEPc. */

  /* Finalize PETSc and MPI. */
  PetscCall(PetscFinalize());

  return 0;
}
