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


PetscErrorCode ComputeConstantRHS(KSP ksp, Vec b, void *user)
{
  Context      *ctx=(Context *)user;
  PetscScalar  Kx, Ky, Kz;
  PetscReal    detA;
  DM           vlasovDM=ctx->vlasovDM;
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

  // Extract the density vector.
  PetscCall(GetFieldVec(vlasovDM, ctx->vlasov, "density", &density));

  // Set the RHS vector equal to the global mean density.
  PetscCall(VecMean(density, &mean));
  val = mean * (2.0 * dx*dy*dz / detA);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mean = %f\n", mean));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "val = %f\n", val));
  PetscCall(VecSet(b, val));

  // Restore the density vector.
  PetscCall(RestoreFieldVec(vlasovDM, ctx->vlasov, "density", &density));

  // Store the RHS vector in the problem context.
  ctx->rhs = b;

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode ComputeSinusoidalRHS(KSP ksp, Vec b, void *user)
{
  Context      *ctx=(Context *)user;
  PetscScalar  Kx, Ky, Kz;
  PetscReal    detA;
  PetscReal    dx=ctx->grid.d.x;
  PetscReal    dy=ctx->grid.d.y;
  PetscReal    dz=ctx->grid.d.z;
  DM           vlasovDM=ctx->vlasovDM;
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

  // Zero the incoming vector.
  PetscCall(VecZeroEntries(b));

  // Get the DM associated with the KSP.
  PetscCall(KSPGetDM(ksp, &dm));

  // Get an array equivalent to the RHS Vec.
  PetscCall(DMDAVecGetArray(dm, b, &rhs));

  // Extract the density vector and compute its mean.
  PetscCall(GetFieldVec(vlasovDM, ctx->vlasov, "density", &density));
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
  PetscCall(RestoreFieldVec(vlasovDM, ctx->vlasov, "density", &density));

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


PetscErrorCode ComputeFullRHS(KSP ksp, Vec b, void *user)
{
  // the problem context
  Context      *ctx=(Context *)user;
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
  // geometric scale factors
  PetscScalar  hx, hy, hz;
  PetscScalar  hxx, hyx, hzx, hxy, hyy, hzy, hxz, hyz, hzz;
  // global scale factor
  PetscReal    scale;
  // the DM of the grid
  DM           vlasovDM=ctx->vlasovDM;
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
  PetscInt     im1, ip1, jm1, jp1, km1, kp1;
  PetscInt     Nx=ctx->grid.N.x;
  PetscInt     Ny=ctx->grid.N.y;
  PetscInt     Nz=ctx->grid.N.z;
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
  PetscScalar  Eterm=0.0, Pterm=0.0, Gterm=0.0;
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

  // Compute geometric scale factors.
  hx  = 1.0 / (2.0 * dx);
  hy  = 1.0 / (2.0 * dy);
  hz  = 1.0 / (2.0 * dz);
  hxx = 4.0 * hx*hx;
  hyy = 4.0 * hy*hy;
  hzz = 4.0 * hz*hz;
  hxy = hyx = hx*hy;
  hxz = hzx = hx*hz;
  hzy = hyz = hz*hy;

  // Assign the global scale factor. Separating this out makes it easier to
  // rescale the operator, for example while debugging.
  //
  // TODO: Ensure that we scale LHS and RHS values by the same factor, possibly
  // by storing `scale` in the context.
  scale = 2.0 * dx*dy*dz;

  // Extract the density array.
  PetscCall(DMGetLocalVector(vlasovDM, &gridvec));
  PetscCall(DMGlobalToLocalBegin(vlasovDM, ctx->vlasov, INSERT_VALUES, gridvec));
  PetscCall(DMGlobalToLocalEnd(vlasovDM, ctx->vlasov, INSERT_VALUES, gridvec));
  PetscCall(DMDAVecGetArray(vlasovDM, gridvec, &gridarr));

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

        // Define backward and forward indices. [DEV] Use periodic BC for now.
        im1 = (i == 0) ? Nx-1 : i-1;
        ip1 = (i == Nx-1) ? 0 : i+1;
        jm1 = (j == 0) ? Ny-1 : j-1;
        jp1 = (j == Ny-1) ? 0 : j+1;
        km1 = (k == 0) ? Nz-1 : k-1;
        kp1 = (k == Nz-1) ? 0 : k+1;

        // Assign temporary density values.
        nijk = gridarr[k][j][i].n;
        nmjk = gridarr[k][j][im1].n;
        npjk = gridarr[k][j][ip1].n;
        nimk = gridarr[k][jm1][i].n;
        nipk = gridarr[k][jp1][i].n;
        nijm = gridarr[km1][j][i].n;
        nijp = gridarr[kp1][j][i].n;
        nppk = gridarr[k][jp1][ip1].n;
        npmk = gridarr[k][jm1][ip1].n;
        nmpk = gridarr[k][jp1][im1].n;
        nmmk = gridarr[k][jm1][im1].n;
        npjp = gridarr[kp1][j][ip1].n;
        npjm = gridarr[km1][j][ip1].n;
        nmjp = gridarr[kp1][j][im1].n;
        nmjm = gridarr[km1][j][im1].n;
        nipp = gridarr[kp1][jp1][i].n;
        nipm = gridarr[km1][jp1][i].n;
        nimp = gridarr[kp1][jm1][i].n;
        nimm = gridarr[km1][jm1][i].n;

        /* Assign the RHS value at (i, j, k). */
        Eterm = // div(n R E0)
          (rxx*E0x + rxy*E0y + rxz*E0z)*(npjk - nmjk)*hx +
          (ryx*E0x + ryy*E0y + ryz*E0z)*(nipk - nimk)*hy +
          (rzx*E0x + rzy*E0y + rzz*E0z)*(nijp - nijm)*hz;
        Pterm = // div(R div(P)) / e
          cth * (
            hxx * rxx * (npjk - 2.0*nijk + nmjk) +
            hxy * rxy * (nppk - npmk - nmpk + nmmk) +
            hxz * rxz * (npjp - npjm - nmjp + nmjm) +
            hyx * ryx * (nppk - npmk - nmpk + nmmk) +
            hyy * ryy * (nipk - 2.0*nijk + nimk) +
            hyz * ryz * (nipp - nipm - nimp + nimm) +
            hzx * rzx * (npjp - npjm - nmjp + nmjm) +
            hzy * rzy * (nipp - nipm - nimp + nimm) +
            hzz * rzz * (nijp - 2.0*nijk + nijm));
        Gterm = // (1+kappa^2) (me nue / e) div(flux)
          detA * cG * (
            (gridarr[k][j][ip1].flux[0] - gridarr[k][j][im1].flux[0])*hx +
            (gridarr[k][jp1][i].flux[1] - gridarr[k][jm1][i].flux[1])*hy +
            (gridarr[kp1][j][i].flux[2] - gridarr[km1][j][i].flux[2])*hz);
        rhs[k][j][i] = scale * (Eterm + Pterm + Gterm);
      }
    }
  }

  // Restore the borrowed objects.
  PetscCall(DMDAVecRestoreArray(vlasovDM, gridvec, &gridarr));
  PetscCall(DMRestoreLocalVector(vlasovDM, &gridvec));
  PetscCall(DMDAVecRestoreArray(dm, b, &rhs));

  // Make the RHS vector consistent with the LHS operator.
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  // Store the RHS vector in the problem context.
  ctx->rhs = b;

  PetscFunctionReturn(PETSC_SUCCESS);
}


