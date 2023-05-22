#include <petsc.h>
#include "hybrid.h"
#include "lhs.h"

PetscErrorCode ComputeIdentityLHS(KSP ksp, Mat J, Mat A, void *_ctx)
{
  Context      *ctx=(Context *)_ctx;

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
PetscErrorCode ComputeLaplacianLHS(KSP ksp, Mat J, Mat A, void *_ctx)
{
  // the problem context
  Context      *ctx=(Context *)_ctx;
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


PetscErrorCode ComputeFullLHS(KSP ksp, Mat J, Mat A, void *_ctx)
{
  // the problem context
  Context      *ctx=(Context *)_ctx;
  // components of magnetization vector
  PetscScalar  Kx, Ky, Kz;
  // matrix determinant
  PetscReal    detA;
  // components of magnetization tensor
  PetscScalar  rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz;
  // geometric scale factors
  PetscScalar  sxx, syx, szx, sxy, syy, szy, sxz, syz, szz;
  // the DM of the grid
  DM           grid;
  // local grid vector
  Vec          gridvec;
  // array representation of grid quantities
  GridNode  ***array;
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

  // Get the grid DM from the context.
  PetscCall(DMSwarmGetCellDM(ctx->swarm, &grid));

  // Extract the density array.
  PetscCall(DMGetLocalVector(grid, &gridvec));
  PetscCall(DMGlobalToLocalBegin(grid, ctx->vlasov, INSERT_VALUES, gridvec));
  PetscCall(DMGlobalToLocalEnd(grid, ctx->vlasov, INSERT_VALUES, gridvec));
  PetscCall(DMDAVecGetArray(grid, gridvec, &array));

  // Get the DM associated with the KSP.
  PetscCall(KSPGetDM(ksp, &dm));

  // Get this processor's indices.
  PetscCall(DMDAGetCorners(dm, &i0, &j0, &k0, &ni, &nj, &nk));

  // Loop over grid points.
  for (k=k0; k<k0+nk; k++) {
    for (j=j0; j<j0+nj; j++) {
      for (i=i0; i<i0+ni; i++) {

        // Define forward and backward indices.

        // Assign density values
        nijk = array[k][j][i].n;
        nmjk = array[k][j][i-1].n;
        npjk = array[k][j][i+1].n;
        nimk = array[k][j-1][i].n;
        nipk = array[k][j+1][i].n;
        nijm = array[k-1][j][i].n;
        nijp = array[k+1][j][i].n;

        /* x-y corner coefficients */
        vppk =  sxy*rxy*(npjk + nijk) + syx*ryx*(nipk + nijk);
        vpmk = -sxy*rxy*(npjk + nijk) - syx*ryx*(nijk + nimk);
        vmpk = -sxy*rxy*(nijk + nmjk) - syx*ryx*(nipk + nijk);
        vmmk =  sxy*rxy*(nijk + nmjk) + syx*ryx*(nijk + nimk);
        /* x-z corner coefficients */
        vpjp =  sxz*rxz*(npjk + nijk) + szx*rzx*(nijp + nijk);
        vpjm = -sxz*rxz*(npjk + nijk) - szx*rzx*(nijk + nijm);
        vmjp = -sxz*rxz*(nijk + nmjk) - szx*rzx*(nijp + nijk);
        vmjm =  sxz*rxz*(nijk + nmjk) + szx*rzx*(nijk + nijm);
        /* y-z corner coefficients */
        vipp =  syz*ryz*(nipk + nijk) + szy*rzy*(nijp + nijk);
        vipm = -syz*ryz*(nipk + nijk) - szy*rzy*(nijk + nijm);
        vimp = -syz*ryz*(nijk + nimk) - szy*rzy*(nijp + nijk);
        vimm =  syz*ryz*(nijk + nimk) + szy*rzy*(nijk + nijm);
        /* star-stencil coefficients */
        vpjk =  sxx*rxx*(npjk + nijk) + vppk + vpmk + vpjp + vpjm;
        vmjk =  sxx*rxx*(nijk + nmjk) + vmpk + vmmk + vmjp + vmjm;
        vipk =  syy*ryy*(nipk + nijk) + vppk + vmpk + vipp + vipm;
        vimk =  syy*ryy*(nijk + nimk) + vpmk + vmmk + vimp + vimm;
        vijp =  szz*rzz*(nijp + nijk) + vpjp + vmjp + vipp + vimp;
        vijm =  szz*rzz*(nijk + nijm) + vpjm + vmjm + vipm + vimm;
        /* diagonal coefficient */
        vijk = -(vpjk + vipk + vijp + vmjk + vimk + vijm);

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

  PetscCall(DMDAVecRestoreArray(grid, gridvec, &array));
  PetscCall(DMRestoreLocalVector(grid, &gridvec));

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


