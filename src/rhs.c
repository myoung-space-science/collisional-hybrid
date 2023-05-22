#include <petsc.h>
#include "hybrid.h"
#include "rhs.h"

/* Get the global vector corresponding to a named DM field.

  Note: Something like this may exist in PETSc but I can't find it.
*/
static PetscErrorCode
GetFieldVec(DM dm, Vec full, const char *name, Vec *vec)
{
  PetscInt  nf;
  char      **names;
  DM        *dmArray;
  PetscInt  field;
  PetscBool found;

  PetscFunctionBeginUser;

  PetscCall(DMCreateFieldDecomposition(dm, &nf, &names, NULL, &dmArray));
  for (field=0; field<nf; field++) {
    PetscCall(PetscStrcmp(name, names[field], &found));
    if (found) {
      PetscCall(DMGetGlobalVector(dmArray[field], vec));
      PetscCall(VecStrideGather(full, field, *vec, INSERT_VALUES));
      PetscCall(PetscObjectSetName((PetscObject)*vec, names[field]));
      break;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Restore a vector retrieved via GetFieldVec.

  Note: Something like this may exist in PETSc but I can't find it.
*/
static PetscErrorCode
RestoreFieldVec(DM dm, Vec full, const char *name, Vec *vec)
{
  PetscInt  nf;
  char      **names;
  DM        *dmArray;
  PetscInt  field;
  PetscBool found;

  PetscFunctionBeginUser;

  PetscCall(DMCreateFieldDecomposition(dm, &nf, &names, NULL, &dmArray));
  for (field=0; field<nf; field++) {
    PetscCall(PetscStrcmp(name, names[field], &found));
    if (found) {
      PetscCall(DMRestoreGlobalVector(dmArray[field], vec));
      break;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode ComputeConstantRHS(KSP ksp, Vec b, void *_ctx)
{
  Context      *ctx=(Context *)_ctx;
  PetscScalar  Kx, Ky, Kz;
  PetscReal    detA;
  DM           grid;
  PetscReal    dx=ctx->grid.d.x;
  PetscReal    dy=ctx->grid.d.y;
  PetscReal    dz=ctx->grid.d.z;
  Vec          density;
  PetscScalar  mean, val;

  PetscFunctionBeginUser;

  // Extract values of electron magnetization.
  Kx = ctx->electrons.kappa.x;
  Ky = ctx->electrons.kappa.y;
  Kz = ctx->electrons.kappa.z;

  // Compute the value of the matrix determinant.
  detA = 1 + Kx*Kx + Ky*Ky + Kz*Kz;

  // Get the grid DM from the context.
  PetscCall(DMSwarmGetCellDM(ctx->swarm, &grid));

  // Extract the density vector.
  PetscCall(GetFieldVec(grid, ctx->vlasov, "density", &density));

  // Set the RHS vector equal to the global mean density.
  PetscCall(VecMean(density, &mean));
  val = mean * (2.0 * dx*dy*dz / detA);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mean = %f\n", mean));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "val = %f\n", val));
  PetscCall(VecSet(b, val));

  // Restore the density vector.
  PetscCall(RestoreFieldVec(grid, ctx->vlasov, "density", &density));

  // Store the RHS vector in the problem context.
  ctx->rhs = b;

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode ComputeSinusoidalRHS(KSP ksp, Vec b, void *_ctx)
{
  Context      *ctx=(Context *)_ctx;
  PetscScalar  Kx, Ky, Kz;
  PetscReal    detA;
  PetscReal    dx=ctx->grid.d.x;
  PetscReal    dy=ctx->grid.d.y;
  PetscReal    dz=ctx->grid.d.z;
  DM           grid;
  Vec          density;
  PetscScalar  n0;
  DM           dm;
  PetscScalar  ***rhs;
  PetscInt     i0, j0, k0;
  PetscInt     ni, nj, nk;
  PetscInt     i, j, k;
  PetscScalar  x, y, z;
  PetscScalar  Cx, Cy, Cz;
  PetscScalar  val;
  MatNullSpace nullspace;

  PetscFunctionBeginUser;

  // Extract values of electron magnetization.
  Kx = ctx->electrons.kappa.x;
  Ky = ctx->electrons.kappa.y;
  Kz = ctx->electrons.kappa.z;

  // Compute the value of the matrix determinant.
  detA = 1 + Kx*Kx + Ky*Ky + Kz*Kz;

  // Get the grid DM from the context.
  PetscCall(DMSwarmGetCellDM(ctx->swarm, &grid));

  // Zero the incoming vector.
  PetscCall(VecZeroEntries(b));

  // Get the DM associated with the KSP.
  PetscCall(KSPGetDM(ksp, &dm));

  // Get an array equivalent to the RHS Vec.
  PetscCall(DMDAVecGetArray(dm, b, &rhs));

  // Extract the density vector and compute its mean.
  PetscCall(GetFieldVec(grid, ctx->vlasov, "density", &density));
  PetscCall(VecMean(density, &n0));

  // Get this processor's indices.
  PetscCall(DMDAGetCorners(dm, &i0, &j0, &k0, &ni, &nj, &nk));

  // Loop over grid points.
  for (k=k0; k<k0+nk; k++) {
    for (j=j0; j<j0+nj; j++) {
      for (i=i0; i<i0+ni; i++) {
        x = ((PetscReal)i + 0.5)*dx;
        y = ((PetscReal)j + 0.5)*dy;
        z = ((PetscReal)k + 0.5)*dz;
        Cx = PetscCosScalar(2*PETSC_PI * x);
        Cy = PetscCosScalar(2*PETSC_PI * y);
        Cz = PetscCosScalar(2*PETSC_PI * z);
        val = Cx * Cy * Cz;
        rhs[k][j][i] = n0 * val * (2.0 * dx*dy*dz / detA);
      }
    }
  }

  // Restore the density vector.
  PetscCall(RestoreFieldVec(grid, ctx->vlasov, "density", &density));

  // Restore the borrowed arrays.
  PetscCall(DMDAVecRestoreArray(dm, b, &rhs));

  // Make the RHS vector consistent with the LHS operator.
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  // Store the RHS vector in the problem context.
  ctx->rhs = b;

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode ComputeFullRHS(KSP ksp, Vec b, void *_ctx)
{
  // the problem context
  Context      *ctx=(Context *)_ctx;
  // components of magnetization vector
  PetscScalar  Kx, Ky, Kz;
  // matrix determinant
  PetscReal    detA;
  // components of magnetization tensor
  PetscScalar  rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz;
  // x-axis cell spacing
  PetscReal    dx=ctx->grid.d.x;
  // y-axis cell spacing
  PetscReal    dy=ctx->grid.d.y;
  // z-axis cell spacing
  PetscReal    dz=ctx->grid.d.z;
  // the DM of the grid
  DM           grid;
  // local grid vector
  Vec          gridvec;
  // array representation of grid quantities
  GridNode     ***gridarr;
  // the DM of the KSP
  DM           dm;
  // array of RHS values
  PetscScalar  ***rhs;
  // indices of the lower left corner of the local grid
  PetscInt     i0, j0, k0;
  // number of non-ghost cells in each dimension of the local grid
  PetscInt     ni, nj, nk;
  // grid indices
  PetscInt     i, j, k;
  // the density value at the current and neighboring grid points
  PetscScalar  nijk, npjk, nmjk, nipk, nimk, nijp, nijm;
  PetscScalar  nppk, npmk, nmpk, nmmk;
  PetscScalar  npjp, npjm, nmjp, nmjm;
  PetscScalar  nipp, nipm, nimp, nimm;
  PetscScalar  E0x=ctx->plasma.E0.x;
  PetscScalar  E0y=ctx->plasma.E0.y;
  PetscScalar  E0z=ctx->plasma.E0.z;
  PetscScalar  cth=ctx->electrons.gamma*KB*ctx->electrons.T/Q;
  PetscScalar  cG=ctx->electrons.m*ctx->electrons.nu/Q;
  PetscScalar  val;
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

  // Get the grid DM from the context.
  PetscCall(DMSwarmGetCellDM(ctx->swarm, &grid));

  // Extract the density array.
  PetscCall(DMGetLocalVector(grid, &gridvec));
  PetscCall(DMGlobalToLocalBegin(grid, ctx->vlasov, INSERT_VALUES, gridvec));
  PetscCall(DMGlobalToLocalEnd(grid, ctx->vlasov, INSERT_VALUES, gridvec));
  PetscCall(DMDAVecGetArray(grid, gridvec, &gridarr));

  // Zero the incoming vector.
  PetscCall(VecZeroEntries(b));

  // Get the DM associated with the KSP.
  PetscCall(KSPGetDM(ksp, &dm));

  // Get an array equivalent to the RHS Vec.
  PetscCall(DMDAVecGetArray(dm, b, &rhs));

  // Get this processor's indices.
  PetscCall(DMDAGetCorners(dm, &i0, &j0, &k0, &ni, &nj, &nk));

  // Loop over grid points.
  for (k=k0; k<k0+nk; k++) {
    for (j=j0; j<j0+nj; j++) {
      for (i=i0; i<i0+ni; i++) {

        // Assign temporary density values.
        nijk = gridarr[k][j][i].n;
        nmjk = gridarr[k][j][i-1].n;
        npjk = gridarr[k][j][i+1].n;
        nimk = gridarr[k][j-1][i].n;
        nipk = gridarr[k][j+1][i].n;
        nijm = gridarr[k-1][j][i].n;
        nijp = gridarr[k+1][j][i].n;
        nppk = gridarr[k][j+1][i+1].n;
        npmk = gridarr[k][j-1][i+1].n;
        nmpk = gridarr[k][j+1][i-1].n;
        nmmk = gridarr[k][j-1][i-1].n;
        npjp = gridarr[k+1][j][i+1].n;
        npjm = gridarr[k-1][j][i+1].n;
        nmjp = gridarr[k+1][j][i-1].n;
        nmjm = gridarr[k-1][j][i-1].n;
        nipp = gridarr[k+1][j+1][i].n;
        nipm = gridarr[k-1][j+1][i].n;
        nimp = gridarr[k+1][j-1][i].n;
        nimm = gridarr[k-1][j-1][i].n;

        // Assign the RHS value at (i, j, k).
        val =
          // div(n R E0)
          (rxx*E0x + rxy*E0y + rxz*E0z)*(npjk - nmjk)*dy*dz +
          (ryx*E0x + ryy*E0y + ryz*E0z)*(nipk - nimk)*dy*dz +
          (rzx*E0x + rzy*E0y + rzz*E0z)*(nijp - nijm)*dy*dz +
          // div(R div(P)) / e
          cth * (
            rxx * (npjk - 2.0*nijk + nmjk) * (2.0*dy*dz/dx) +
            rxy * (nppk - npmk - nmpk + nmmk) * (0.5*dz) +
            rxz * (npjp - npjm - nmjp + nmjm) * (0.5*dy) +
            ryx * (nppk - npmk - nmpk + nmmk) * (0.5*dz) +
            ryy * (nipk - 2.0*nijk + nimk) * (2.0*dx*dz/dy) +
            ryz * (nipp - nipm - nimp + nimm) * (0.5*dx) +
            rzx * (npjp - npjm - nmjp + nmjm) * (0.5*dy) +
            rzy * (nipp - nipm - nimp + nimm) * (0.5*dx) +
            rzz * (nijp - 2.0*nijk + nijm) * (2.0*dx*dy/dz)) +
          // (1+kappa^2) (me nue / e) div(flux)
          detA * cG * (
            (gridarr[k][j][i+1].flux[0]-gridarr[k][j][i-1].flux[0])*(dy*dz) +
            (gridarr[k][j+1][i].flux[1]-gridarr[k][j-1][i].flux[1])*(dx*dz) +
            (gridarr[k+1][j][i].flux[2]-gridarr[k-1][j][i].flux[2])*(dx*dy));
        rhs[k][j][i] = val / detA;
      }
    }
  }

  // Restore the borrowed arrays.
  PetscCall(DMDAVecRestoreArray(grid, gridvec, &gridarr));
  PetscCall(DMDAVecRestoreArray(dm, b, &rhs));

  // Make the RHS vector consistent with the LHS operator.
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  // Store the RHS vector in the problem context.
  ctx->rhs = b;

  PetscFunctionReturn(PETSC_SUCCESS);
}

