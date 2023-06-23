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

  // Assign the global scale factor. Separating this out makes it easier to
  // rescale the operator, for example while debugging.
  //
  // TODO: Ensure that we scale LHS and RHS values by the same factor, possibly
  // by storing `scale` in the context.
  scale = 1.0 / detA;

  // get density and flux arrays.
  PetscCall(GetLocalVlasovField("density", &density, &n, ctx));
  PetscCall(GetLocalVlasovField("x flux", &xflux, &Gx, ctx));
  PetscCall(GetLocalVlasovField("y flux", &yflux, &Gy, ctx));
  PetscCall(GetLocalVlasovField("z flux", &zflux, &Gz, ctx));

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

        // Compute local derivatives.
        dndx   = 0.0; dndy   = 0.0; dndz   = 0.0;
        dGxdx  = 0.0; dGydy  = 0.0; dGzdz  = 0.0;
        d2ndxx = 0.0; d2ndxy = 0.0; d2ndxz = 0.0;
        d2ndyx = 0.0; d2ndyy = 0.0; d2ndyz = 0.0;
        d2ndzx = 0.0; d2ndzy = 0.0; d2ndzz = 0.0;
        PetscCall(dFdx(  n, dx,     i, j, k, &dndx,   CENTERED));
        PetscCall(dFdy(  n, dy,     i, j, k, &dndy,   CENTERED));
        PetscCall(dFdz(  n, dz,     i, j, k, &dndz,   CENTERED));
        PetscCall(dFdx( Gx, dx,     i, j, k, &dGxdx,  CENTERED));
        PetscCall(dFdy( Gy, dy,     i, j, k, &dGydy,  CENTERED));
        PetscCall(dFdz( Gz, dz,     i, j, k, &dGzdz,  CENTERED));
        PetscCall(d2Fdxx(n, dx,     i, j, k, &d2ndxx, CENTERED));
        PetscCall(d2Fdxy(n, dx, dy, i, j, k, &d2ndxy, CENTERED, CENTERED));
        PetscCall(d2Fdxz(n, dx, dz, i, j, k, &d2ndxz, CENTERED, CENTERED));
        PetscCall(d2Fdyx(n, dy, dx, i, j, k, &d2ndyx, CENTERED, CENTERED));
        PetscCall(d2Fdyy(n, dy,     i, j, k, &d2ndyy, CENTERED));
        PetscCall(d2Fdyz(n, dy, dz, i, j, k, &d2ndyz, CENTERED, CENTERED));
        PetscCall(d2Fdzx(n, dz, dx, i, j, k, &d2ndzx, CENTERED, CENTERED));
        PetscCall(d2Fdzy(n, dz, dy, i, j, k, &d2ndzy, CENTERED, CENTERED));
        PetscCall(d2Fdzz(n, dz,     i, j, k, &d2ndzz, CENTERED));

        /* Assign the RHS value at (i, j, k). */
        Eterm = // div(n R E0)
          (rxx*E0x + rxy*E0y + rxz*E0z)*dndx +
          (ryx*E0x + ryy*E0y + ryz*E0z)*dndy +
          (rzx*E0x + rzy*E0y + rzz*E0z)*dndz;
        Pterm = // (me vTe / e) div(R div(P))
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

  // Restore density and flux arrays.
  PetscCall(RestoreLocalVlasovField("density", &density, &n, ctx));
  PetscCall(RestoreLocalVlasovField("x flux", &xflux, &Gx, ctx));
  PetscCall(RestoreLocalVlasovField("y flux", &yflux, &Gy, ctx));
  PetscCall(RestoreLocalVlasovField("z flux", &zflux, &Gz, ctx));

  // Make the RHS vector consistent with the LHS operator.
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  // Store the RHS vector in the problem context.
  ctx->rhs = b;

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


