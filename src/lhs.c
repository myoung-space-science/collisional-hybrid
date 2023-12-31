#include <petsc.h>
#include <slepceps.h>
#include "constants.h"
#include "hybrid.h"
#include "lhs.h"

PetscErrorCode ComputeInteriorStencil(PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, MatStencil cols[NVALUES], PetscReal vals[NVALUES])
{
  PetscFunctionBeginUser;
  // Assign the value at node (i+1, j, k).
  vals[0] = f[k][j][i+1];
  cols[0].i = i+1;
  cols[0].j = j;
  cols[0].k = k;
  // Assign the value at node (i-1, j, k).
  vals[1] = f[k][j][i-1];
  cols[1].i = i-1;
  cols[1].j = j;
  cols[1].k = k;
  // Assign the value at node (i, j+1, k).
  vals[2] = f[k][j+1][i];
  cols[2].i = i;
  cols[2].j = j+1;
  cols[2].k = k;
  // Assign the value at node (i, j-1, k).
  vals[3] = f[k][j-1][i];
  cols[3].i = i;
  cols[3].j = j-1;
  cols[3].k = k;
  // Assign the value at node (i, j, k+1).
  vals[4] = f[k+1][j][i];
  cols[4].i = i;
  cols[4].j = j;
  cols[4].k = k+1;
  // Assign the value at node (i, j, k-1).
  vals[5] = f[k-1][j][i];
  cols[5].i = i;
  cols[5].j = j;
  cols[5].k = k-1;
  // Assign the value at node (i+1, j+1, k).
  vals[6] = f[k][j+1][i+1];
  cols[6].i = i+1;
  cols[6].j = j+1;
  cols[6].k = k;
  // Assign the value at node (i+1, j-1, k).
  vals[7] = f[k][j-1][i+1];
  cols[7].i = i+1;
  cols[7].j = j-1;
  cols[7].k = k;
  // Assign the value at node (i-1, j+1, k).
  vals[8] = f[k][j+1][i-1];
  cols[8].i = i-1;
  cols[8].j = j+1;
  cols[8].k = k;
  // Assign the value at node (i-1, j-1, k).
  vals[9] = f[k][j-1][i-1];
  cols[9].i = i-1;
  cols[9].j = j-1;
  cols[9].k = k;
  // Assign the value at node (i+1, j, k+1).
  vals[10] = f[k+1][j][i+1];
  cols[10].i = i+1;
  cols[10].j = j;
  cols[10].k = k+1;
  // Assign the value at node (i+1, j, k-1).
  vals[11] = f[k-1][j][i+1];
  cols[11].i = i+1;
  cols[11].j = j;
  cols[11].k = k-1;
  // Assign the value at node (i-1, j, k+1).
  vals[12] = f[k+1][j][i-1];
  cols[12].i = i-1;
  cols[12].j = j;
  cols[12].k = k+1;
  // Assign the value at node (i-1, j, k-1).
  vals[13] = f[k-1][j][i-1];
  cols[13].i = i-1;
  cols[13].j = j;
  cols[13].k = k-1;
  // Assign the value at node (i, j+1, k+1).
  vals[14] = f[k+1][j+1][i];
  cols[14].i = i;
  cols[14].j = j+1;
  cols[14].k = k+1;
  // Assign the value at node (i, j+1, k-1).
  vals[15] = f[k-1][j+1][i];
  cols[15].i = i;
  cols[15].j = j+1;
  cols[15].k = k-1;
  // Assign the value at node (i, j-1, k+1).
  vals[16] = f[k+1][j-1][i];
  cols[16].i = i;
  cols[16].j = j-1;
  cols[16].k = k+1;
  // Assign the value at node (i, j-1, k-1).
  vals[17] = f[k-1][j-1][i];
  cols[17].i = i;
  cols[17].j = j-1;
  cols[17].k = k-1;
  // Assign the value at node (i, j, k).
  vals[18] = f[k][j][i];
  cols[18].i = i;
  cols[18].j = j;
  cols[18].k = k;
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode ComputePeriodicStencil(PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, MatStencil cols[NVALUES], PetscReal vals[NVALUES], void *ctx)
{
  PetscFunctionBeginUser;

  PetscCall(ComputeInteriorStencil(i, j, k, f, cols, vals));

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode ComputeNeumannStencil(PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, MatStencil cols[NVALUES], PetscReal vals[NVALUES], void *ctx)
{
  Context  *user=(Context *)ctx;
  PetscInt Nx=user->grid.N.x;
  PetscInt Ny=user->grid.N.y;
  PetscInt Nz=user->grid.N.z;

  PetscFunctionBeginUser;

  // TODO
  if (i==0 || j==0 || k==0 || i==Nx-1 || j==Ny-1 || k==Nz-1) {
    if (i == Nx-1) {}
    if (i == 0) {}
    if (j == Ny-1) {}
    if (j == 0) {}
    if (k == Nz-1) {}
    if (k == 0) {}
  } else {
    PetscCall(ComputeInteriorStencil(i, j, k, f, cols, vals));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode ComputeIdentityLHS(KSP ksp, Mat J, Mat A, void *user)
{
  Context      *ctx=(Context *)user;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  PetscCall(MatZeroEntries(A));
  PetscCall(MatShift(A, 1.0));

  if (ctx->viewLHS) {
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "lhs.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(MatView(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


// Note: This is more complex than necessary because it arose as a
// simplification of ComputeFullLHS. It could be further simplified.
PetscErrorCode ComputeLaplacianLHS(KSP ksp, Mat J, Mat A, void *user)
{
  // the problem context
  Context      *ctx=(Context *)user;
  // geometric scale factors
  PetscScalar  sxx, syy, szz;
  // the DM of the KSP
  DM           dm;
  // indices of the lower left corner of the local grid
  PetscInt     i0, j0, k0;
  // number of non-ghost cells in each dimension of the local grid
  PetscInt     ni, nj, nk;
  // grid indices
  PetscInt     i, j, k;
  // diagonal coefficient
  PetscScalar  vijk=1.0;
  // star-stencil coefficients
  PetscScalar  vpjk=0.0, vmjk=0.0, vipk=0.0, vimk=0.0, vijp=0.0, vijm=0.0;
  // x-y corners
  PetscScalar  vmmk=0.0, vpmk=0.0, vmpk=0.0, vppk=0.0;
  // x-z corners
  PetscScalar  vpjp=0.0, vpjm=0.0, vmjp=0.0, vmjm=0.0;
  // y-z corners
  PetscScalar  vipp=0.0, vipm=0.0, vimp=0.0, vimm=0.0;
  // the current value at each active stencil point
  PetscScalar  val[NVALUES];
  // the current matrix row
  MatStencil   row;
  // the current matrix column of each active stencil point
  MatStencil   col[NVALUES];
  // the operator nullspace
  MatNullSpace nullspace;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Compute constant stencil values.
  sxx = ctx->grid.d.y * ctx->grid.d.z / ctx->grid.d.x;
  syy = ctx->grid.d.x * ctx->grid.d.z / ctx->grid.d.y;
  szz = ctx->grid.d.x * ctx->grid.d.y / ctx->grid.d.z;

  // Assign the star-stencil coefficients.
  vpjk =  sxx;
  vmjk =  sxx;
  vipk =  syy;
  vimk =  syy;
  vijp =  szz;
  vijm =  szz;

  // Assign the diagonal coefficient.
  vijk = -(vpjk + vipk + vijp + vmjk + vimk + vijm);

  // Get the DM associated with the KSP.
  PetscCall(KSPGetDM(ksp, &dm));

  // Get this processor's indices.
  PetscCall(DMDAGetCorners(dm, &i0, &j0, &k0, &ni, &nj, &nk));

  // Loop over grid points.
  for (k=k0; k<k0+nk; k++) {
    for (j=j0; j<j0+nj; j++) {
      for (i=i0; i<i0+ni; i++) {
        row.i = i; row.j = j; row.k = k;
        // Interior node (i+1, j, k)
        val[0] = vpjk;
        col[0].i = i+1;
        col[0].j = j;
        col[0].k = k;
        // Interior node (i-1, j, k)
        val[1] = vmjk;
        col[1].i = i-1;
        col[1].j = j;
        col[1].k = k;
        // Interior node (i, j+1, k)
        val[2] = vipk;
        col[2].i = i;
        col[2].j = j+1;
        col[2].k = k;
        // Interior node (i, j-1, k)
        val[3] = vimk;
        col[3].i = i;
        col[3].j = j-1;
        col[3].k = k;
        // Interior node (i, j, k+1)
        val[4] = vijp;
        col[4].i = i;
        col[4].j = j;
        col[4].k = k+1;
        // Interior node (i, j, k-1)
        val[5] = vijm;
        col[5].i = i;
        col[5].j = j;
        col[5].k = k-1;
        // Interior node (i+1, j+1, k)
        val[6] = vppk;
        col[6].i = i+1;
        col[6].j = j+1;
        col[6].k = k;
        // Interior node (i+1, j-1, k)
        val[7] = vpmk;
        col[7].i = i+1;
        col[7].j = j-1;
        col[7].k = k;
        // Interior node (i-1, j+1, k)
        val[8] = vmpk;
        col[8].i = i-1;
        col[8].j = j+1;
        col[8].k = k;
        // Interior node (i-1, j-1, k)
        val[9] = vmmk;
        col[9].i = i-1;
        col[9].j = j-1;
        col[9].k = k;
        // Interior node (i+1, j, k+1)
        val[10] = vpjp;
        col[10].i = i+1;
        col[10].j = j;
        col[10].k = k+1;
        // Interior node (i+1, j, k-1)
        val[11] = vpjm;
        col[11].i = i+1;
        col[11].j = j;
        col[11].k = k-1;
        // Interior node (i-1, j, k+1)
        val[12] = vmjp;
        col[12].i = i-1;
        col[12].j = j;
        col[12].k = k+1;
        // Interior node (i-1, j, k-1)
        val[13] = vmjm;
        col[13].i = i-1;
        col[13].j = j;
        col[13].k = k-1;
        // Interior node (i, j+1, k+1)
        val[14] = vipp;
        col[14].i = i;
        col[14].j = j+1;
        col[14].k = k+1;
        // Interior node (i, j+1, k-1)
        val[15] = vipm;
        col[15].i = i;
        col[15].j = j+1;
        col[15].k = k-1;
        // Interior node (i, j-1, k+1)
        val[16] = vimp;
        col[16].i = i;
        col[16].j = j-1;
        col[16].k = k+1;
        // Interior node (i, j-1, k-1)
        val[17] = vimm;
        col[17].i = i;
        col[17].j = j-1;
        col[17].k = k-1;
        // Interior node (i, j, k)
        val[18] = vijk;
        col[18].i = row.i;
        col[18].j = row.j;
        col[18].k = row.k;
        PetscCall(MatSetValuesStencil(A, 1, &row, NVALUES, col, val, INSERT_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(A, nullspace));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  if (ctx->viewLHS) {
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "lhs.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(MatView(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode ComputeFullLHS(KSP ksp, Mat J, Mat A, void *user)
{
  // the problem context
  Context      *ctx=(Context *)user;
  // components of magnetization vector
  PetscScalar  Kx, Ky, Kz;
  // matrix determinant
  PetscReal    detA;
  // cell sizes
  PetscReal     dx=ctx->grid.d.x;
  PetscReal     dy=ctx->grid.d.y;
  PetscReal     dz=ctx->grid.d.z;
  // components of magnetization tensor
  PetscScalar  rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz;
  // geometric scale factors
  PetscScalar  hxx, hyx, hzx, hxy, hyy, hzy, hxz, hyz, hzz;
  // global scale factor
  PetscScalar  scale;
  // coefficient scale factors
  PetscScalar  sxx, syx, szx, sxy, syy, szy, sxz, syz, szz;
  // the DM of the grid
  DM           vlasovDM=ctx->vlasovDM;
  // local grid vector
  Vec          gridvec;
  // array representation of grid quantities
  GridNode     ***array;
  // the DM of the KSP
  DM           dm;
  // indices of the lower left corner of the local grid
  PetscInt     i0, j0, k0;
  // number of non-ghost cells in each dimension of the local grid
  PetscInt     ni, nj, nk;
  // grid indices
  PetscInt     i, j, k;
  PetscInt     im1, ip1, jm1, jp1, km1, kp1;
  // the density value at the current and neighboring grid points
  PetscScalar  nijk, npjk, nmjk, nipk, nimk, nijp, nijm;
  // discretization coefficients
  Vec          F;
  PetscReal    ***f;
  // the stencil function
  StencilFunc  ComputeStencil=ctx->stencilFunc;
  // the current value at each active stencil point
  PetscScalar  vals[NVALUES];
  // the current matrix row
  MatStencil   row;
  // the current matrix column of each active stencil point
  MatStencil   cols[NVALUES];
  // the operator nullspace
  MatNullSpace nullspace;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Extract values of electron magnetization.
  Kx = ctx->electrons.kappa.x;
  Ky = ctx->electrons.kappa.y;
  Kz = ctx->electrons.kappa.z;

  // Compute the value of the matrix determinant.
  detA = 1 + Kx*Kx + Ky*Ky + Kz*Kz;

  // Define components of the magnetization tensor.
  rxx = 1 + Kx*Kx;
  rxy = Ky*Kx - Kz;
  rxz = Kz*Kx + Ky;
  ryx = Kx*Ky + Kz;
  ryy = 1 + Ky*Ky;
  ryz = Kz*Ky - Kx;
  rzx = Kx*Kz - Ky;
  rzy = Ky*Kz + Kx;
  rzz = 1 + Kz*Kz;

  // Compute geometric scale factors for stencil values.
  hxx = 1.0 / (2.0 * dx*dx);
  hyy = 1.0 / (2.0 * dy*dy);
  hzz = 1.0 / (2.0 * dz*dz);
  hxy = hyx = 1.0 / (8.0 * dy*dx);
  hxz = hzx = 1.0 / (8.0 * dz*dx);
  hyz = hzy = 1.0 / (8.0 * dy*dz);

  // Assign the global scale factor. Separating this out makes it easier to
  // rescale the operator, for example while debugging.
  //
  // TODO: Ensure that we scale LHS and RHS values by the same factor, possibly
  // by storing `scale` in the context.
  scale = 1.0 / detA;

  // Compute coefficient scale factors;
  sxx = scale * rxx*hxx;
  syy = scale * ryy*hyy;
  szz = scale * rzz*hzz;
  sxy = scale * rxy*hxy;
  sxz = scale * rxz*hxz;
  syz = scale * ryz*hyz;
  syx = scale * ryx*hyx;
  szx = scale * rzx*hzx;
  szy = scale * rzy*hzy;

  // Extract the density array.
  PetscCall(DMGetLocalVector(vlasovDM, &gridvec));
  PetscCall(DMGlobalToLocalBegin(vlasovDM, ctx->vlasov, INSERT_VALUES, gridvec));
  PetscCall(DMGlobalToLocalEnd(vlasovDM, ctx->vlasov, INSERT_VALUES, gridvec));
  PetscCall(DMDAVecGetArray(vlasovDM, gridvec, &array));

  // Get the DM associated with the KSP.
  PetscCall(KSPGetDM(ksp, &dm));

  // Get a local array of zeros that shares data with the KSP DM.
  PetscCall(DMGetLocalVector(dm, &F));
  PetscCall(VecZeroEntries(F));
  PetscCall(DMDAVecGetArray(dm, F, &f));

  // Get this processor's indices.
  PetscCall(DMDAGetCorners(dm, &i0, &j0, &k0, &ni, &nj, &nk));

  // Loop over grid points.
  for (k=k0; k<k0+nk; k++) {
    for (j=j0; j<j0+nj; j++) {
      for (i=i0; i<i0+ni; i++) {

        // Define backward and forward indices.
        im1 = i-1;
        ip1 = i+1;
        jm1 = j-1;
        jp1 = j+1;
        km1 = k-1;
        kp1 = k+1;

        // Assign density values.
        nijk = array[k][j][i].n;
        nmjk = array[k][j][im1].n;
        npjk = array[k][j][ip1].n;
        nimk = array[k][jm1][i].n;
        nipk = array[k][jp1][i].n;
        nijm = array[km1][j][i].n;
        nijp = array[kp1][j][i].n;

        /* x-y corner coefficients */
        f[k][j+1][i+1] =  sxy*(npjk + nijk) + syx*(nipk + nijk);
        f[k][j-1][i+1] = -sxy*(npjk + nijk) - syx*(nijk + nimk);
        f[k][j+1][i-1] = -sxy*(nijk + nmjk) - syx*(nipk + nijk);
        f[k][j-1][i-1] =  sxy*(nijk + nmjk) + syx*(nijk + nimk);
        /* x-z corner coefficients */
        f[k+1][j][i+1] =  sxz*(npjk + nijk) + szx*(nijp + nijk);
        f[k-1][j][i+1] = -sxz*(npjk + nijk) - szx*(nijk + nijm);
        f[k+1][j][i-1] = -sxz*(nijk + nmjk) - szx*(nijp + nijk);
        f[k-1][j][i-1] =  sxz*(nijk + nmjk) + szx*(nijk + nijm);
        /* y-z corner coefficients */
        f[k+1][j+1][i] =  syz*(nipk + nijk) + szy*(nijp + nijk);
        f[k-1][j+1][i] = -syz*(nipk + nijk) - szy*(nijk + nijm);
        f[k+1][j-1][i] = -syz*(nijk + nimk) - szy*(nijp + nijk);
        f[k-1][j-1][i] =  syz*(nijk + nimk) + szy*(nijk + nijm);
        /* star-stencil coefficients */
        f[k][j][i+1] =  sxx*(npjk + nijk) + syx*(nipk - nimk) + szx*(nijp - nijm);
        f[k][j][i-1] =  sxx*(nijk + nmjk) - syx*(nipk - nimk) - szx*(nijp - nijm);
        f[k][j+1][i] =  syy*(nipk + nijk) + sxy*(npjk - nmjk) + szy*(nijp - nijm);
        f[k][j-1][i] =  syy*(nijk + nimk) - sxy*(npjk - nmjk) - szy*(nijp - nijm);
        f[k+1][j][i] =  szz*(nijp + nijk) + sxz*(npjk - nmjk) + syz*(nipk - nimk);
        f[k-1][j][i] =  szz*(nijk + nijm) - sxz*(npjk - nmjk) - syz*(nipk - nimk);
        /* diagonal coefficient */
        f[k][j][i] = -(sxx*(npjk + 2*nijk + nmjk) + syy*(nipk + 2*nijk + nimk) + szz*(nijp + 2*nijk + nijm));

        // Compute the stencil values.
        PetscCall(ComputeStencil(i, j, k, f, cols, vals, (void *)ctx));
        row.i = i; row.j = j; row.k = k;
        PetscCall(MatSetValuesStencil(A, 1, &row, NVALUES, cols, vals, INSERT_VALUES));
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(dm, F, &f));
  PetscCall(DMRestoreLocalVector(dm, &F));

  PetscCall(DMDAVecRestoreArray(vlasovDM, gridvec, &array));
  PetscCall(DMRestoreLocalVector(vlasovDM, &gridvec));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(A, nullspace));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  if (ctx->viewLHS) {
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "lhs.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(MatView(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute the eigenvalues of the operator matrix.

This is essentially a distilled version of ${SLEPC_DIR}/src/eps/tutorials/ex1.c
*/
PetscErrorCode ComputeLHSEigenvalues(KSP ksp)
{
  EPS         eps;
  EPSType     type;
  Mat         A;
  PetscInt    i, its, maxit, nev, nconv;
  PetscReal   tol, error, re, im;
  PetscScalar kr, ki;
  Vec         xr, xi;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  /* Set up eigenvalue solver. */
  PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));

  /* Get the LHS operator from the Krylov context. */
  PetscCall(KSPGetOperators(ksp, &A, NULL));

  /* Create vectors for holding the real and imaginary parts. */
  PetscCall(MatCreateVecs(A, NULL, &xr));
  PetscCall(MatCreateVecs(A, NULL, &xi));

  /* Associate the LHS operator with the eigenvalue solver. */
  PetscCall(EPSSetOperators(eps, A, NULL));

  /* Set up the problem type and read CLI options. */
  PetscCall(EPSSetProblemType(eps, EPS_HEP));
  PetscCall(EPSSetFromOptions(eps));

  /* Compute eigenvalues. */
  PetscCall(EPSSolve(eps));

  /* Print results. */
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

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}