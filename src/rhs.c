#include <petsc.h>
#include "calculus.h"
#include "hybrid.h"
#include "rhs.h"
#include "particles.h"


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
  ECHO_FUNCTION_ENTER;

  // Extract values of electron magnetization.
  Kx = ctx->electrons.kappa.x;
  Ky = ctx->electrons.kappa.y;
  Kz = ctx->electrons.kappa.z;

  // Compute the value of the matrix determinant.
  detA = 1 + Kx*Kx + Ky*Ky + Kz*Kz;

  // Extract the density vector.
  PetscCall(GetGlobalVlasovField("density", &density, NULL, ctx));

  // Set the RHS vector equal to the global mean density.
  PetscCall(VecMean(density, &mean));
  val = mean * (2.0 * dx*dy*dz / detA);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mean = %f\n", mean));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "val = %f\n", val));
  PetscCall(VecSet(b, val));

  // Restore the density vector.
  PetscCall(RestoreGlobalVlasovField("density", &density, NULL, ctx));

  // Store the RHS vector in the problem context.
  ctx->rhs = b;

  ECHO_FUNCTION_EXIT;
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
  ECHO_FUNCTION_ENTER;

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
  PetscCall(GetGlobalVlasovField("density", &density, NULL, ctx));
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
  PetscCall(RestoreGlobalVlasovField("density", &density, NULL, ctx));

  // Restore the borrowed arrays.
  PetscCall(DMDAVecRestoreArray(dm, b, &rhs));

  // Make the RHS vector consistent with the LHS operator.
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  // Store the RHS vector in the problem context.
  ctx->rhs = b;

  ECHO_FUNCTION_EXIT;
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
  // x-axis number of cells
  PetscInt     Nx=ctx->grid.N.x;
  // y-axis number of cells
  PetscInt     Ny=ctx->grid.N.y;
  // z-axis number of cells
  PetscInt     Nz=ctx->grid.N.z;
  // geometric scale factors
  PetscScalar  hx, hy, hz;
  PetscScalar  hxx, hyx, hzx, hxy, hyy, hzy, hxz, hyz, hzz;
  // global scale factor
  PetscReal    scale;
  // the DM of the grid
  DM           vlasovDM=ctx->vlasovDM;
  DM           *dms;
  PetscInt     field, nf;
  char         **names;
  // full vector of vlasov quantities
  Vec          vlasov=ctx->vlasov;
  // temporary vectors for a single vlasov component
  Vec          gtmp;
  // local vlasov vectors
  Vec          density, xflux, yflux, zflux;
  // array representations of vlasov quantities
  PetscReal    ***n, ***Gx, ***Gy, ***Gz;
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
  // the density value at the current and neighboring grid points
  PetscScalar  nijk, npjk, nmjk, nipk, nimk, nijp, nijm;
  PetscScalar  nppk, npmk, nmpk, nmmk;
  PetscScalar  npjp, npjm, nmjp, nmjm;
  PetscScalar  nipp, nipm, nimp, nimm;
  DifferenceType xDiffType, yDiffType, zDiffType;
  // first partial density derivatives
  PetscReal    dndx, dndy, dndz;
  // first partial flux derivatives
  PetscReal    dGxdx, dGydy, dGzdz;
  // second partial density derivatives
  PetscReal    d2ndxx, d2ndyy, d2ndzz, d2ndxy, d2ndyx, d2ndxz, d2ndzx, d2ndyz, d2ndzy;
  PetscScalar  E0x=ctx->plasma.E0.x;
  PetscScalar  E0y=ctx->plasma.E0.y;
  PetscScalar  E0z=ctx->plasma.E0.z;
  PetscScalar  cth=ctx->electrons.gamma*KB*ctx->electrons.T/Q;
  PetscScalar  cG=ctx->electrons.m*ctx->electrons.nu/Q;
  PetscScalar  Eterm=0.0, Pterm=0.0, Gterm=0.0;
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

  // Extract density and flux arrays.
  PetscCall(DMCreateFieldDecomposition(vlasovDM, &nf, &names, NULL, &dms));

  PetscCall(DMGetGlobalVector(dms[0], &gtmp));
  PetscCall(VecStrideGather(vlasov, 0, gtmp, INSERT_VALUES));
  PetscCall(DMGetLocalVector(dms[0], &density));
  PetscCall(DMGlobalToLocalBegin(dms[0], gtmp, INSERT_VALUES, density));
  PetscCall(DMGlobalToLocalEnd(dms[0], gtmp, INSERT_VALUES, density));
  PetscCall(DMDAVecGetArray(dms[0], density, &n));
  PetscCall(DMRestoreGlobalVector(dms[0], &gtmp));

  PetscCall(DMGetGlobalVector(dms[1], &gtmp));
  PetscCall(VecStrideGather(vlasov, 1, gtmp, INSERT_VALUES));
  PetscCall(DMGetLocalVector(dms[1], &xflux));
  PetscCall(DMGlobalToLocalBegin(dms[1], gtmp, INSERT_VALUES, xflux));
  PetscCall(DMGlobalToLocalEnd(dms[1], gtmp, INSERT_VALUES, xflux));
  PetscCall(DMDAVecGetArray(dms[1], xflux, &Gx));
  PetscCall(DMRestoreGlobalVector(dms[1], &gtmp));

  PetscCall(DMGetGlobalVector(dms[2], &gtmp));
  PetscCall(VecStrideGather(vlasov, 2, gtmp, INSERT_VALUES));
  PetscCall(DMGetLocalVector(dms[2], &yflux));
  PetscCall(DMGlobalToLocalBegin(dms[2], gtmp, INSERT_VALUES, yflux));
  PetscCall(DMGlobalToLocalEnd(dms[2], gtmp, INSERT_VALUES, yflux));
  PetscCall(DMDAVecGetArray(dms[2], yflux, &Gy));
  PetscCall(DMRestoreGlobalVector(dms[2], &gtmp));

  PetscCall(DMGetGlobalVector(dms[3], &gtmp));
  PetscCall(VecStrideGather(vlasov, 3, gtmp, INSERT_VALUES));
  PetscCall(DMGetLocalVector(dms[3], &zflux));
  PetscCall(DMGlobalToLocalBegin(dms[3], gtmp, INSERT_VALUES, zflux));
  PetscCall(DMGlobalToLocalEnd(dms[3], gtmp, INSERT_VALUES, zflux));
  PetscCall(DMDAVecGetArray(dms[3], zflux, &Gz));
  PetscCall(DMRestoreGlobalVector(dms[3], &gtmp));

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

        // Define backward and forward indices.
        im1 = i-1;
        ip1 = i+1;
        jm1 = j-1;
        jp1 = j+1;
        km1 = k-1;
        kp1 = k+1;

        // Assign temporary density values.
        nijk = n[k][j][i];
        nmjk = n[k][j][im1];
        npjk = n[k][j][ip1];
        nimk = n[k][jm1][i];
        nipk = n[k][jp1][i];
        nijm = n[km1][j][i];
        nijp = n[kp1][j][i];
        nppk = n[k][jp1][ip1];
        npmk = n[k][jm1][ip1];
        nmpk = n[k][jp1][im1];
        nmmk = n[k][jm1][im1];
        npjp = n[kp1][j][ip1];
        npjm = n[km1][j][ip1];
        nmjp = n[kp1][j][im1];
        nmjm = n[km1][j][im1];
        nipp = n[kp1][jp1][i];
        nipm = n[km1][jp1][i];
        nimp = n[kp1][jm1][i];
        nimm = n[km1][jm1][i];

        // Compute local derivatives.
        dndx = 0.0;
        dndy = 0.0;
        dndz = 0.0;
        dGxdx = 0.0;
        dGydy = 0.0;
        dGzdz = 0.0;
        d2ndxx = 0.0;
        d2ndyy = 0.0;
        d2ndzz = 0.0;
        d2ndxy = 0.0;
        d2ndxz = 0.0;
        d2ndyx = 0.0;
        d2ndyz = 0.0;
        d2ndzx = 0.0;
        d2ndzy = 0.0;
        if (i == 0) {
          xDiffType = FORWARD;
        } else if (i == Nx-1) {
          xDiffType = BACKWARD;
        } else {
          xDiffType = CENTERED;
        }
        if (j == 0) {
          yDiffType = FORWARD;
        } else if (j == Ny-1) {
          yDiffType = BACKWARD;
        } else {
          yDiffType = CENTERED;
        }
        if (k == 0) {
          zDiffType = FORWARD;
        } else if (k == Nz-1) {
          zDiffType = BACKWARD;
        } else {
          zDiffType = CENTERED;
        }
        PetscCall(dFdx(  n, dx, i, j, k, &dndx,   xDiffType));
        PetscCall(dFdy(  n, dy, i, j, k, &dndy,   yDiffType));
        PetscCall(dFdz(  n, dz, i, j, k, &dndz,   zDiffType));
        PetscCall(dFdx( Gx, dx, i, j, k, &dGxdx,  xDiffType));
        PetscCall(dFdy( Gy, dy, i, j, k, &dGydy,  yDiffType));
        PetscCall(dFdz( Gz, dz, i, j, k, &dGzdz,  zDiffType));
        PetscCall(d2Fdxx(n, dx, i, j, k, &d2ndxx, xDiffType));
        PetscCall(d2Fdyy(n, dy, i, j, k, &d2ndyy, yDiffType));
        PetscCall(d2Fdzz(n, dz, i, j, k, &d2ndzz, zDiffType));
        PetscCall(d2Fdxy(n, dx, dy, i, j, k, &d2ndxy, CENTERED, CENTERED));
        PetscCall(d2Fdxz(n, dx, dz, i, j, k, &d2ndxz, CENTERED, CENTERED));
        PetscCall(d2Fdyx(n, dy, dx, i, j, k, &d2ndyx, CENTERED, CENTERED));
        PetscCall(d2Fdyz(n, dy, dz, i, j, k, &d2ndyz, CENTERED, CENTERED));
        PetscCall(d2Fdzx(n, dz, dx, i, j, k, &d2ndzx, CENTERED, CENTERED));
        PetscCall(d2Fdzy(n, dz, dy, i, j, k, &d2ndzy, CENTERED, CENTERED));

        /* Assign the RHS value at (i, j, k). */
        Eterm = // div(n R E0)
          (rxx*E0x + rxy*E0y + rxz*E0z)*dndx +
          (ryx*E0x + ryy*E0y + ryz*E0z)*dndy +
          (rzx*E0x + rzy*E0y + rzz*E0z)*dndz;
        Pterm = // div(R div(P)) / e
          cth * (
            rxx*d2ndxx + rxy*d2ndxy + rxz*d2ndxz +
            ryx*d2ndyx + ryy*d2ndyy + ryz*d2ndyz +
            rzx*d2ndzx + rzy*d2ndzy + rzz*d2ndzz);
        Gterm = // (1+kappa^2) (me nue / e) div(flux)
          detA * cG * (dGxdx + dGydy + dGzdz);
        rhs[k][j][i] = scale * (Eterm + Pterm + Gterm);
      }
    }
  }

  // Restore the borrowed objects.
  PetscCall(DMDAVecRestoreArray(dm, b, &rhs));
  PetscCall(DMDAVecRestoreArray(dms[0], density, &n));
  PetscCall(DMRestoreLocalVector(dms[0], &density));
  PetscCall(DMDAVecRestoreArray(dms[1], xflux, &Gx));
  PetscCall(DMRestoreLocalVector(dms[1], &xflux));
  PetscCall(DMDAVecRestoreArray(dms[2], yflux, &Gy));
  PetscCall(DMRestoreLocalVector(dms[2], &yflux));
  PetscCall(DMDAVecRestoreArray(dms[3], zflux, &Gz));
  PetscCall(DMRestoreLocalVector(dms[3], &zflux));

  // Release memory.
  for (field=0; field<nf; field++) {
    PetscFree(names[field]);
    PetscCall(DMDestroy(&dms[field]));
  }
  PetscFree(names);
  PetscFree(dms);
  PetscCall(DMDestroy(&vlasovDM));

  // Make the RHS vector consistent with the LHS operator.
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  // Store the RHS vector in the problem context.
  ctx->rhs = b;

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


