#include <petsc.h>
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
  vals[12] = f[k+1][k][i-1];
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

  PetscCall(MatZeroEntries(A));
  PetscCall(MatShift(A, 1.0));

  if (ctx->viewLHS) {
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "lhs.dat", FILE_MODE_WRITE, &viewer));
    PetscCall(MatView(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

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
  // components of magnetization tensor
  PetscScalar  rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz;
  // geometric scale factors
  PetscScalar  sxx, syx, szx, sxy, syy, szy, sxz, syz, szz;
  // the DM of the grid
  DM           vlasovDM;
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

  /* Compute geometric scale factors for stencil values. Note that there is some
  redundancy for the sake of organization.

  Diagonal factors have the form
  - `sii = 2*di*dj*dk / (2*di^2*|A|) = dj*dk / (di*|A|)`

  Off-diagonal factors have the form
  - `sij = 2*di*dj*dk / (8*di*dj*|A|) = dk / (4*|A|)`
  */
  sxx = ctx->grid.d.y * ctx->grid.d.z / ctx->grid.d.x / detA;
  syx = 0.25*ctx->grid.d.z  / detA;
  szx = 0.25*ctx->grid.d.y  / detA;
  sxy = 0.25*ctx->grid.d.z  / detA;
  syy = ctx->grid.d.x * ctx->grid.d.z / ctx->grid.d.y / detA;
  szy = 0.25*ctx->grid.d.x  / detA;
  sxz = 0.25*ctx->grid.d.y  / detA;
  syz = 0.25*ctx->grid.d.x  / detA;
  szz = ctx->grid.d.x * ctx->grid.d.y / ctx->grid.d.z / detA;

  // TODO: Pre-compute sij*rij coefficients for efficiency.

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

        // Assign density values.
        nijk = array[k][j][i].n;
        nmjk = array[k][j][i-1].n;
        npjk = array[k][j][i+1].n;
        nimk = array[k][j-1][i].n;
        nipk = array[k][j+1][i].n;
        nijm = array[k-1][j][i].n;
        nijp = array[k+1][j][i].n;

        /* x-y corner coefficients */
        f[k][j+1][i+1] =  sxy*rxy*(npjk + nijk) + syx*ryx*(nipk + nijk);
        f[k][j-1][i+1] = -sxy*rxy*(npjk + nijk) - syx*ryx*(nijk + nimk);
        f[k][j+1][i-1] = -sxy*rxy*(nijk + nmjk) - syx*ryx*(nipk + nijk);
        f[k][j-1][i-1] =  sxy*rxy*(nijk + nmjk) + syx*ryx*(nijk + nimk);
        /* x-z corner coefficients */
        f[k+1][j][i+1] =  sxz*rxz*(npjk + nijk) + szx*rzx*(nijp + nijk);
        f[k-1][j][i+1] = -sxz*rxz*(npjk + nijk) - szx*rzx*(nijk + nijm);
        f[k+1][j][i-1] = -sxz*rxz*(nijk + nmjk) - szx*rzx*(nijp + nijk);
        f[k-1][j][i-1] =  sxz*rxz*(nijk + nmjk) + szx*rzx*(nijk + nijm);
        /* y-z corner coefficients */
        f[k+1][j+1][i] =  syz*ryz*(nipk + nijk) + szy*rzy*(nijp + nijk);
        f[k-1][j+1][i] = -syz*ryz*(nipk + nijk) - szy*rzy*(nijk + nijm);
        f[k+1][j-1][i] = -syz*ryz*(nijk + nimk) - szy*rzy*(nijp + nijk);
        f[k-1][j-1][i] =  syz*ryz*(nijk + nimk) + szy*rzy*(nijk + nijm);
        /* star-stencil coefficients */
        f[k][j][i+1] =  sxx*rxx*(npjk + nijk) + syx*ryx*(nipk - nimk) + szx*rzx*(nijp - nijm);
        f[k][j][i-1] =  sxx*rxx*(nijk + nmjk) - syx*ryx*(nipk - nimk) - szx*rzx*(nijp - nijm);
        f[k][j+1][i] =  syy*ryy*(nipk + nijk) + sxy*rxy*(npjk - nmjk) + szy*rzy*(nijp - nijm);
        f[k][j-1][i] =  syy*ryy*(nijk + nimk) - sxy*rxy*(npjk - nmjk) - szy*rzy*(nijp - nijm);
        f[k+1][j][i] =  szz*rzz*(nijp + nijk) + sxz*rxz*(npjk - nmjk) + syz*ryz*(nipk - nimk);
        f[k-1][j][i] =  szz*rzz*(nijk + nijm) - sxz*rxz*(npjk - nmjk) - syz*ryz*(nipk - nimk);
        /* diagonal coefficient */
        f[k][j][i] = -(sxx*rxx*(npjk + 2*nijk + nmjk) + syy*ryy*(nipk + 2*nijk + nimk) + szz*rzz*(nijp + 2*nijk + nijm));

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

  PetscFunctionReturn(PETSC_SUCCESS);
}


