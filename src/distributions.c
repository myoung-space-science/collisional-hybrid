#include <petsc.h>
#include "hybrid.h"
#include "random.h"
#include "distributions.h"


PetscErrorCode UniformDistribution_FromSwarm(Context *ctx)
{
  DM          ionsDM=ctx->ionsDM;
  PetscReal   min[NDIM], max[NDIM];
  PetscInt    npoints[NDIM];

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  min[0] = ctx->grid.p0.x;
  min[1] = ctx->grid.p0.y;
  min[2] = ctx->grid.p0.z;
  max[0] = ctx->grid.p1.x - ctx->grid.d.x;
  max[1] = ctx->grid.p1.y - ctx->grid.d.y;
  max[2] = ctx->grid.p1.z - ctx->grid.d.z;
  npoints[0] = 4 * ctx->grid.N.x;
  npoints[1] = 4 * ctx->grid.N.y;
  npoints[2] = 4 * ctx->grid.N.z;

  PetscCall(DMSwarmSetPointsUniformCoordinates(ionsDM, min, max, npoints, INSERT_VALUES));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode UniformDistribution(Context *ctx)
{
  DM          ionsDM=ctx->ionsDM;
  PetscScalar *coords;
  PetscInt    np, np_cell, ip;
  DM          vlasov;
  PetscInt    i0, j0, k0;
  PetscInt    ni, nj, nk, nc;
  PetscInt    i, j, k, idx;
  PetscInt    dim;
  PetscReal   r[NDIM];
  PetscReal   d[NDIM]={ctx->grid.d.x, ctx->grid.d.y, ctx->grid.d.z};

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get information about the discrete grid.
  PetscCall(DMSwarmGetCellDM(ionsDM, &vlasov));
  PetscCall(DMDAGetCorners(vlasov, &i0, &j0, &k0, &ni, &nj, &nk));

  // Get the local number of ions.
  PetscCall(DMSwarmGetLocalSize(ionsDM, &np));

  // Compute the number of ions per cell. Note that np_cell*nc will not in
  // general be equal to the input value of -Np, if given.
  nc = ni*nj*nk;
  np_cell = (PetscInt)(np / nc);

  // Reset the local swarm size to avoid a seg fault when accessing the
  // coordinates array. Passing a negative value for the buffer forces the swarm
  // to use its existing buffer size.
  PetscCall(DMSwarmSetLocalSizes(ionsDM, np_cell*nc, -1));

  // Get a representation of the particle coordinates.
  PetscCall(DMSwarmGetField(ionsDM, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Loop over cells and place an equal number of particles at the center of
  // each cell in all but the last row of each index. This will result in a
  // symmetric step-function distribution. It is not truly uniform.
  for (ip=0; ip<np_cell; ip++) {
    for (i=i0; i<i0+ni-1; i++) {
      for (j=j0; j<j0+nj-1; j++) {
        for (k=k0; k<k0+nk-1; k++) {
          idx = (ip*nc + k + j*nk + i*nk*nj)*NDIM;
          r[0] = (PetscReal)(i + 1);
          r[1] = (PetscReal)(j + 1);
          r[2] = (PetscReal)(k + 1);
          for (dim=0; dim<NDIM; dim++) {
            coords[idx + dim] = d[dim]*(r[dim] - 0.5);
          }
        }
      }
    }
  }

  // Restore the coordinates array.
  PetscCall(DMSwarmRestoreField(ionsDM, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SobolDistribution(Context *ctx)
{
  DM          ionsDM=ctx->ionsDM;
  PetscInt    seed=-1, ndim=NDIM;
  PetscReal   *coords, v;
  PetscInt    np, ip;
  PetscReal   r[NDIM];
  PetscReal   L[NDIM]={ctx->grid.L.x, ctx->grid.L.y, ctx->grid.L.z};
  PetscReal   d[NDIM]={ctx->grid.d.x, ctx->grid.d.y, ctx->grid.d.z};
  PetscReal   p0[NDIM]={ctx->grid.p0.x, ctx->grid.p0.y, ctx->grid.p0.z};
  PetscReal   p1[NDIM]={ctx->grid.p1.x, ctx->grid.p1.y, ctx->grid.p1.z};
  PetscInt    dim;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get a representation of the particle coordinates.
  PetscCall(DMSwarmGetField(ionsDM, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Initialize the psuedo-random number generator.
  PetscCall(Sobseq(&seed, r-1));

  // Get the local number of particles.
  PetscCall(DMSwarmGetLocalSize(ionsDM, &np));

  for (ip=0; ip<np; ip++) {
    PetscCall(Sobseq(&ndim, r-1));
    for (dim=0; dim<NDIM; dim++) {
      v = r[dim]*(L[dim] + d[dim]) - 0.5*d[dim];
      if (v < p0[dim]) {
        coords[ip*NDIM + dim] = p0[dim] + 0.01*d[dim];
      } else if (v > p1[dim]) {
        coords[ip*NDIM + dim] = p1[dim] - 0.01*d[dim];
      } else {
        coords[ip*NDIM + dim] = v;
      }
    }
  }

  // Restore the coordinates array.
  PetscCall(DMSwarmRestoreField(ionsDM, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode SinusoidalDistribution(PetscReal x, PetscReal y, PetscReal z, PetscReal *v, Context *ctx)
{
  PetscReal fx;

  PetscFunctionBeginUser;

  fx = PetscSinReal(2*PETSC_PI * x);
  *v = 1.0 + 0.25*fx;

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode Rejection(DistributionFunction density, Context *ctx)
{
  PetscRandom random;
  DM          ionsDM=ctx->ionsDM;
  PetscInt    np, ip;
  DM          vlasov;
  PetscInt    i0, ni, i;
  PetscInt    j0, nj, j;
  PetscInt    k0, nk, k;
  PetscReal   localMax=0.0;
  PetscScalar *coords;
  PetscReal   Lx=ctx->grid.L.x;
  PetscReal   Ly=ctx->grid.L.y;
  PetscReal   Lz=ctx->grid.L.z;
  PetscReal   x, y, z, v, w;
  PetscReal   r[NDIM];
  PetscInt    it=0;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get a representation of the ion coordinates.
  PetscCall(DMSwarmGetField(ionsDM, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Create a random number generator.
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &random));
  PetscCall(PetscRandomSetInterval(random, 0.0, 1.0));
  PetscCall(PetscRandomSetSeed(random, ctx->mpi.rank));
  PetscCall(PetscRandomSeed(random));

  // Compute local maximum density.
  PetscCall(DMSwarmGetCellDM(ionsDM, &vlasov));
  PetscCall(DMDAGetCorners(vlasov, &i0, &j0, &k0, &ni, &nj, &nk));
  for (i=i0; i<i0+ni; i++) {
    for (j=j0; j<j0+nj; j++) {
      for (k=k0; k<k0+nk; k++) {
        PetscCall(density(i, j, k, &w, ctx));
        localMax = PetscMax(localMax, w);
      }
    }
  }
  PRINT_RANKS("[%d] Local maximum density: %g\n", ctx->mpi.rank, localMax);

  // Get the local number of ions.
  PetscCall(DMSwarmGetLocalSize(ionsDM, &np));

  // Loop over all local ions.
  ip = 0;
  while (ip < np) {
    PetscCall(PetscRandomGetValuesReal(random, NDIM, r));
    x = r[0] * Lx;
    y = r[1] * Ly;
    z = r[2] * Lz;
    PetscCall(density(x, y, z, &w, ctx));
    PetscCall(PetscRandomGetValueReal(random, &v));
    if (w > v * localMax) {
      coords[ip*NDIM + 0] = x;
      coords[ip*NDIM + 1] = y;
      coords[ip*NDIM + 2] = z;
      ip++;
    }
    it++;
  }

  // Echo rejection efficiency.
  NEWLINE;
  PRINT_RANKS("[%d] Rejection efficiency: %f\n", ctx->mpi.rank, (PetscReal)ip/it);

  // Restore the coordinates array.
  PetscCall(DMSwarmRestoreField(ionsDM, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Destroy the random-number generator.
  PetscCall(PetscRandomDestroy(&random));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


