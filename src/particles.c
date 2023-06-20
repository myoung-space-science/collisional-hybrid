#include <petsc.h>
#include "hybrid.h"
#include "distributions.h"
#include "random.h"
#include "vectors.h"


/* Compute the initial ion positions. */
PetscErrorCode InitializePositions(Context *ctx)
{
  DM          ionsDM=ctx->ionsDM;
  PetscInt    np, Np, ip;
  PetscScalar *coords;
  RealVector  *pos;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Echo sizes.
  PetscCall(DMSwarmGetSize(ionsDM, &Np));
  PetscCall(DMSwarmGetLocalSize(ionsDM, &np));
  NEWLINE;
  PRINT_RANKS("[%d] Local # of ions before placement: %d\n", ctx->mpi.rank, np);
  PRINT_WORLD("   Global # of ions before placement: %d\n", Np);

  // Initialize coordinates in the ions DM.
  switch(ctx->densityType) {
    case DENSITY_FLAT_NORMAL:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Not implemented: %s density", DensityTypes[DENSITY_FLAT_NORMAL]);
      break;
    case DENSITY_FLAT_REVERSE:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Not implemented: %s density", DensityTypes[DENSITY_FLAT_REVERSE]);
      break;
    case DENSITY_FLAT_SOBOL:
      PetscCall(SobolDistribution(ctx));
      break;
    case DENSITY_UNIFORM:
      PetscCall(UniformDistribution(ctx));
      break;
    case DENSITY_SINUSOIDAL:
      PetscCall(Rejection(SinusoidalDistribution, ctx));
      break;
    case DENSITY_GAUSSIAN:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Not implemented: %s density", DensityTypes[DENSITY_GAUSSIAN]);
      break;
  }

  // Echo sizes.
  PetscCall(DMSwarmGetSize(ionsDM, &Np));
  PetscCall(DMSwarmGetLocalSize(ionsDM, &np));
  NEWLINE;
  PRINT_RANKS("[%d] Local # of ions after placement: %d\n", ctx->mpi.rank, np);
  PRINT_WORLD("   Global # of ions after placement: %d\n", Np);

  // Migrate ions between ranks.
  PetscCall(DMSwarmMigrate(ionsDM, PETSC_TRUE));

  // Echo sizes.
  PetscCall(DMSwarmGetSize(ionsDM, &Np));
  PetscCall(DMSwarmGetLocalSize(ionsDM, &np));
  NEWLINE;
  PRINT_RANKS("[%d] Local # of ions after migration: %d\n", ctx->mpi.rank, np);
  PRINT_WORLD("   Global # of ions after migration: %d\n", Np);

  // Get the number of ions on this rank.
  PetscCall(DMSwarmGetLocalSize(ionsDM, &np));

  // Get an array representation of the swarm coordinates.
  PetscCall(DMSwarmGetField(ionsDM, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Get an array representation of the ions positions.
  PetscCall(DMSwarmGetField(ionsDM, "position", NULL, NULL, (void **)&pos));

  // Loop over ions and assign parameter values.
  for (ip=0; ip<np; ip++) {
    pos[ip].x = coords[ip*NDIM + 0];
    pos[ip].y = coords[ip*NDIM + 1];
    pos[ip].z = coords[ip*NDIM + 2];
  }

  // Restore the ion-positions array.
  PetscCall(DMSwarmRestoreField(ionsDM, "position", NULL, NULL, (void **)&pos));

  // Restore the swarm-coordinates array.
  PetscCall(DMSwarmRestoreField(ionsDM, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  // Display information about the ions DM.
  NEWLINE;
  PetscCall(DMView(ionsDM, PETSC_VIEWER_STDOUT_WORLD));

  // Update the parameter context.
  PetscCall(DMSwarmGetSize(ionsDM, &ctx->plasma.Np));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute the initial ion velocities. */
PetscErrorCode InitializeVelocities(Context *ctx)
{
  DM          ionsDM=ctx->ionsDM;
  PetscInt    np, ip;
  RealVector  *vel;
  PetscReal   dvx, dvy, dvz;
  long        seed=ctx->seed;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get the number of local ions.
  PetscCall(DMSwarmGetLocalSize(ionsDM, &np));

  // Get an array representation of the ion velocities.
  PetscCall(DMSwarmGetField(ionsDM, "velocity", NULL, NULL, (void **)&vel));

  // Loop over ions and assign parameter values.
  for (ip=0; ip<np; ip++) {
    PetscCall(Gasdev(&seed, &dvx));
    PetscCall(Gasdev(&seed, &dvy));
    PetscCall(Gasdev(&seed, &dvz));
    vel[ip].x = ctx->ions.vT.x*dvx + ctx->ions.v0.x;
    vel[ip].y = ctx->ions.vT.y*dvy + ctx->ions.v0.y;
    vel[ip].z = ctx->ions.vT.z*dvz + ctx->ions.v0.z;
  }

  // Restore the ion-velocities array.
  PetscCall(DMSwarmRestoreField(ionsDM, "velocity", NULL, NULL, (void **)&vel));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Compute moments of the ion distribution. */
PetscErrorCode CollectVlasovQuantities(Context *ctx)
{
  DM          vlasovDM, ionsDM=ctx->ionsDM;
  Vec         gridvec;
  GridNode    ***array;
  PetscInt    dim;
  PetscReal   x, y, z, dx, dy, dz;
  RealVector  r, *pos, v, *vel;
  PetscInt    ip, np;
  PetscInt    ixl, ixh, iyl, iyh, izl, izh;
  PetscReal   wxl, wxh, wyl, wyh, wzl, wzh;
  PetscReal   hhh, lhh, hlh, llh, hhl, lhl, hll, lll;
  PetscReal   w[NDIM];

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get the vlasov DM from the ions DM.
  PetscCall(DMSwarmGetCellDM(ionsDM, &vlasovDM));

  // Get a vector for the local portion of the grid.
  PetscCall(DMGetLocalVector(vlasovDM, &gridvec));

  // Make sure the local grid vector has zeroes everywhere.
  PetscCall(VecZeroEntries(gridvec));

  // Get an array corresponding to the local vlasov quantities. Creating this
  // array from the local vector causes it to have room for the ghost points.
  PetscCall(DMDAVecGetArray(vlasovDM, gridvec, &array));

  // Get an array representation of the ion positions.
  PetscCall(DMSwarmGetField(ionsDM, "position", NULL, NULL, (void **)&pos));

  // Get an array representation of the ion velocities.
  PetscCall(DMSwarmGetField(ionsDM, "velocity", NULL, NULL, (void **)&vel));

  // Get the number of ions on this rank.
  PetscCall(DMSwarmGetLocalSize(ionsDM, &np));

  // Extract cell widths.
  dx = ctx->grid.d.x;
  dy = ctx->grid.d.y;
  dz = ctx->grid.d.z;

  // Loop over ions.
  for (ip=0; ip<np; ip++) {
    // Get the current ion's coordinates.
    r = pos[ip];
    // Normalize each coordinate to a fractional number of grid cells.
    x = r.x / dx;
    y = r.y / dy;
    z = r.z / dz;
    // Compute the x-dimension neighbors and corresponding weights.
    ixl = (PetscInt)x;
    ixh = ixl+1;
    wxh = x - (PetscReal)ixl;
    wxl = 1.0 - wxh;
    // Compute the y-dimension neighbors and corresponding weights.
    iyl = (PetscInt)y;
    iyh = iyl+1;
    wyh = y - (PetscReal)iyl;
    wyl = 1.0 - wyh;
    // Compute the z-dimension neighbors and corresponding weights.
    izl = (PetscInt)z;
    izh = izl+1;
    wzh = z - (PetscReal)izl;
    wzl = 1.0 - wzh;
    // Compute the weight of each nearby grid point.
    hhh = wzh*wyh*wxh;
    lhh = wzl*wyh*wxh;
    hlh = wzh*wyl*wxh;
    llh = wzl*wyl*wxh;
    hhl = wzh*wyh*wxl;
    lhl = wzl*wyh*wxl;
    hll = wzh*wyl*wxl;
    lll = wzl*wyl*wxl;
    // Assign density values (zeroth moment).
    array[izh][iyh][ixh].n += hhh;
    array[izl][iyh][ixh].n += lhh;
    array[izh][iyl][ixh].n += hlh;
    array[izl][iyl][ixh].n += llh;
    array[izh][iyh][ixl].n += hhl;
    array[izl][iyh][ixl].n += lhl;
    array[izh][iyl][ixl].n += hll;
    array[izl][iyl][ixl].n += lll;
    // Assign flux values (first moments wrt velocity).
    v = vel[ip];
    w[0] = v.x;
    w[1] = v.y;
    w[2] = v.z;
    for (dim=0; dim<NDIM; dim++) {
      array[izh][iyh][ixh].flux[dim] += w[dim]*hhh;
      array[izl][iyh][ixh].flux[dim] += w[dim]*lhh;
      array[izh][iyl][ixh].flux[dim] += w[dim]*hlh;
      array[izl][iyl][ixh].flux[dim] += w[dim]*llh;
      array[izh][iyh][ixl].flux[dim] += w[dim]*hhl;
      array[izl][iyh][ixl].flux[dim] += w[dim]*lhl;
      array[izh][iyl][ixl].flux[dim] += w[dim]*hll;
      array[izl][iyl][ixl].flux[dim] += w[dim]*lll;
    }
  }

  // Restore the local grid array.
  PetscCall(DMDAVecRestoreArray(vlasovDM, gridvec, &array));

  // Communicate local information to the persistent global grid vector.
  PetscCall(DMLocalToGlobal(vlasovDM, gridvec, ADD_VALUES, ctx->vlasov));

  // Restore the local grid vector.
  PetscCall(DMRestoreLocalVector(vlasovDM, &gridvec));

  // Restore the ion-positions array.
  PetscCall(DMSwarmRestoreField(ionsDM, "position", NULL, NULL, (void **)&pos));

  // Restore the ion-velocities array.
  PetscCall(DMSwarmRestoreField(ionsDM, "velocity", NULL, NULL, (void **)&vel));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Insert boundary values into Vlasov quantities. */
PetscErrorCode ApplyVlasovBC(Context *ctx)
{
  DM             vlasovDM=ctx->vlasovDM;
  DMBoundaryType xBC=ctx->xDMBC;
  DMBoundaryType yBC=ctx->yDMBC;
  DMBoundaryType zBC=ctx->zDMBC;
  PetscInt       i0, j0, k0;
  PetscInt       ni, nj, nk;
  PetscInt       i, j, k;
  PetscInt       Nx=ctx->grid.N.x;
  PetscInt       Ny=ctx->grid.N.y;
  PetscInt       Nz=ctx->grid.N.z;
  Vec            gridvec;
  GridNode       ***gridarr;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Extract the density and flux arrays.
  PetscCall(DMGetLocalVector(vlasovDM, &gridvec));
  PetscCall(DMGlobalToLocalBegin(vlasovDM, ctx->vlasov, INSERT_VALUES, gridvec));
  PetscCall(DMGlobalToLocalEnd(vlasovDM, ctx->vlasov, INSERT_VALUES, gridvec));
  PetscCall(DMDAVecGetArray(vlasovDM, gridvec, &gridarr));

  // Get this processor's indices.
  PetscCall(DMDAGetCorners(vlasovDM, &i0, &j0, &k0, &ni, &nj, &nk));

  // Insert x-axis boundary values. DEV: periodic.
  for (k=k0; k<k0+nk; k++) {
    for (j=j0; j<j0+nj; j++) {
      // lower boundary
      gridarr[k][j][-1].n       = gridarr[k][j][Nx-1].n;
      gridarr[k][j][-1].flux[0] = gridarr[k][j][Nx-1].flux[0];
      gridarr[k][j][-1].flux[1] = gridarr[k][j][Nx-1].flux[1];
      gridarr[k][j][-1].flux[2] = gridarr[k][j][Nx-1].flux[2];
      // upper boundary
      gridarr[k][j][Nx].n       = gridarr[k][j][0].n;
      gridarr[k][j][Nx].flux[0] = gridarr[k][j][0].flux[0];
      gridarr[k][j][Nx].flux[1] = gridarr[k][j][0].flux[1];
      gridarr[k][j][Nx].flux[2] = gridarr[k][j][0].flux[2];
    }
  }
  // Insert y-axis boundary values. DEV: periodic.
  for (k=k0; k<k0+nk; k++) {
    for (i=i0; i<i0+ni; i++) {
      // lower boundary
      gridarr[k][-1][i].n       = gridarr[k][Ny-1][i].n;
      gridarr[k][-1][i].flux[0] = gridarr[k][Ny-1][i].flux[0];
      gridarr[k][-1][i].flux[1] = gridarr[k][Ny-1][i].flux[1];
      gridarr[k][-1][i].flux[2] = gridarr[k][Ny-1][i].flux[2];
      // upper boundary
      gridarr[k][Ny][i].n       = gridarr[k][0][i].n;
      gridarr[k][Ny][i].flux[0] = gridarr[k][0][i].flux[0];
      gridarr[k][Ny][i].flux[1] = gridarr[k][0][i].flux[1];
      gridarr[k][Ny][i].flux[2] = gridarr[k][0][i].flux[2];
    }
  }
  // Insert z-axis boundary values. DEV: periodic.
  for (j=j0; j<j0+nj; j++) {
    for (i=i0; i<i0+ni; i++) {
      // lower boundary
      gridarr[-1][j][i].n       = gridarr[Nz-1][j][i].n;
      gridarr[-1][j][i].flux[0] = gridarr[Nz-1][j][i].flux[0];
      gridarr[-1][j][i].flux[1] = gridarr[Nz-1][j][i].flux[1];
      gridarr[-1][j][i].flux[2] = gridarr[Nz-1][j][i].flux[2];
      // upper boundary
      gridarr[Nz][j][i].n       = gridarr[0][j][i].n;
      gridarr[Nz][j][i].flux[0] = gridarr[0][j][i].flux[0];
      gridarr[Nz][j][i].flux[1] = gridarr[0][j][i].flux[1];
      gridarr[Nz][j][i].flux[2] = gridarr[0][j][i].flux[2];
    }
  }

  // Restore the borrowed array.
  PetscCall(DMDAVecRestoreArray(vlasovDM, gridvec, &gridarr));

  // Broadcast the values.
  PetscCall(DMLocalToGlobalBegin(vlasovDM, gridvec, INSERT_VALUES, ctx->vlasov));
  PetscCall(DMLocalToGlobalEnd(vlasovDM, gridvec, INSERT_VALUES, ctx->vlasov));

  // Restore the borrowed vector.
  PetscCall(DMRestoreLocalVector(vlasovDM, &gridvec));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Apply the standard Boris-mover algorithm to ion velocities. */
PetscErrorCode BorisMover(KSP ksp, Context *ctx)
{
  PetscReal   q=ctx->ions.q;
  PetscReal   m=ctx->ions.m;
  PetscReal   B[NDIM]={ctx->plasma.B0.x, ctx->plasma.B0.y, ctx->plasma.B0.z};
  PetscReal   dt=ctx->dt;
  PetscInt    dim;
  PetscReal   dx=ctx->grid.d.x, dy=ctx->grid.d.y, dz=ctx->grid.d.z;
  PetscReal   h[NDIM]={1.0/dx, 1.0/dy, 1.0/dz};
  PetscReal   t[NDIM], s[NDIM], t_dot_t;
  PetscReal   tscale, Escale[NDIM];
  DM          ionsDM=ctx->ionsDM;
  DM          phiDM;
  Vec         phiGlobal, phiLocal;
  PetscReal   ***phi;
  PetscInt    np, ip;
  RealVector  r, *pos, v, *vel;
  PetscReal   x, y, z;
  PetscReal   E[NDIM]={0.0, 0.0, 0.0};
  PetscReal   vold[NDIM];
  PetscReal   vminus[NDIM], vprime[NDIM], vplus[NDIM];
  PetscReal   vminus_cross_t[NDIM], vprime_cross_s[NDIM];

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  /* Compute \vec{t} = \frac{q\vec{B}}{m}\frac{\Delta t}{2}. */
  tscale = 0.5 * (q/m) * dt;
  for (dim=0; dim<NDIM; dim++) {
    t[dim] = tscale * B[dim];
  }

  /* Compute \vec{s} = \frac{2\vec{t}}{1 + \vec{t}\cdot\vec{t}}. */
  PetscCall(DotProduct(t, t, &t_dot_t));
  for (dim=0; dim<NDIM; dim++) {
    s[dim] = 2.0 * t[dim] / (1 + t_dot_t);
  }

  /* Compute the electric-field scale factors.

  These account for the species constants as well as the 2nd-order
  finite-difference gradient scale factors.
  */
  for (dim=0; dim<NDIM; dim++) {
    Escale[dim] = -0.5 * h[dim] * tscale;
  }

  /* Get a local copy of phi with ghost cells. */
  PetscCall(KSPGetSolution(ksp, &phiGlobal));
  PetscCall(KSPGetDM(ksp, &phiDM));
  PetscCall(DMGetLocalVector(phiDM, &phiLocal));
  PetscCall(DMGlobalToLocal(phiDM, phiGlobal, INSERT_VALUES, phiLocal));

  /* Get a temporary array representing the local electrostatic potential. */
  PetscCall(DMDAVecGetArray(phiDM, phiLocal, &phi));

  /* Get the number of local ions. */
  PetscCall(DMSwarmGetLocalSize(ionsDM, &np));

  /* Get an array representation of the ion positions. */
  PetscCall(DMSwarmGetField(ionsDM, "position", NULL, NULL, (void **)&pos));

  /* Get an array representation of the ion velocities. */
  PetscCall(DMSwarmGetField(ionsDM, "velocity", NULL, NULL, (void **)&vel));

  /* Loop over particles and interpolate E to grid points. */
  for (ip=0; ip<np; ip++) {
    /* Get the current particle's coordinates. */
    r = pos[ip];
    v = vel[ip];
    /* Normalize each coordinate to a fractional number of grid cells. */
    x = r.x / dx;
    y = r.y / dy;
    z = r.z / dz;
    /* Compute the electric field due to this particle: \vec{E} = -\nabla\phi. */
    PetscCall(DifferenceVector(phi, x, y, z, ctx->grid, E));
    /* Compute \vec{v}^-. */
    vold[0] = v.x;
    vold[1] = v.y;
    vold[2] = v.z;
    for (dim=0; dim<NDIM; dim++) {
      vminus[0] = vold[dim] + Escale[dim]*E[dim];
    }
    /* Compute \vec{v}^- \times \vec{t}. */
    PetscCall(CrossProduct(vminus, t, vminus_cross_t));
    /* Compute \vec{v}^\prime = \vec{v}^- + \vec{v}^- \times \vec{t}. */
    for (dim=0; dim<NDIM; dim++) {
      vprime[dim] = vminus[dim] + vminus_cross_t[dim];
    }
    /* Compute \vec{v}^\prime \times \vec{s}. */
    PetscCall(CrossProduct(vprime, s, vprime_cross_s));
    /* Compute \vec{v}^+ = \vec{v}^- + \vec{v}^\prime \times \vec{s}. */
    for (dim=0; dim<NDIM; dim++) {
      vplus[dim] = vminus[dim] + vprime_cross_s[dim];
    }
    /* Assign new particle velocities. */
    v.x = vplus[0] + Escale[0]*E[0];
    v.y = vplus[1] + Escale[1]*E[1];
    v.z = vplus[2] + Escale[2]*E[2];
    vel[ip] = v;
  }

  // Restore the ion-positions array.
  PetscCall(DMSwarmRestoreField(ionsDM, "position", NULL, NULL, (void **)&pos));

  // Restore the ion-velocities array.
  PetscCall(DMSwarmRestoreField(ionsDM, "velocity", NULL, NULL, (void **)&vel));

  // Restore the borrowed potential array.
  PetscCall(DMDAVecRestoreArray(phiDM, phiLocal, &phi));

  // Restore the borrowed local phi vector.
  PetscCall(DMRestoreLocalVector(phiDM, &phiLocal));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Basic elastic-scattering routine for ion-neutral collisions.

This function is based on, but not identical to, EPPIC elastic_scatter.
*/
PetscErrorCode ComputeCollisions(Context *ctx)
{
  PetscInt   Nc;                                      // the number of collisions to attempt
  PetscInt   Ns=0;                                    // the number of successful collision attempts
  PetscInt   Nf=0;                                    // the number of failed collision attempts
  PetscInt   Np=ctx->plasma.Np;                       // the total number of ions
  PetscInt   ip;                                      // the current ion
  PetscReal  fc=ctx->ions.nu * ctx->dt;               // the product of the collision rate and the time step
  PetscReal  viT=ctx->ions.vT.r;                      // the ion-species thermal speed
  PetscReal  vi0x=ctx->ions.v0.x;                     // the ion-species x-axis drift component
  PetscReal  vi0y=ctx->ions.v0.y;                     // the ion-species y-axis drift component
  PetscReal  vi0z=ctx->ions.v0.z;                     // the ion-species z-axis drift component
  PetscReal  vnT=ctx->neutrals.vT;                    // the neutral-species thermal speed
  PetscReal  vn0=ctx->neutrals.v0.r;                  // the neutral-species drift speed
  PetscReal  vn0x=ctx->neutrals.v0.x;                 // the neutral-species x-axis drift component
  PetscReal  vn0y=ctx->neutrals.v0.y;                 // the neutral-species y-axis drift component
  PetscReal  vn0z=ctx->neutrals.v0.z;                 // the neutral-species z-axis drift component
  PetscReal  vrm;                                     // the maximum ion-neutral relative velocity
  PetscReal  mi=ctx->ions.m;                          // the ion-species mass
  PetscReal  mn=ctx->neutrals.m;                      // the neutral-species mass
  PetscReal  M=mn+mi;                                 // the total mass (mi+mn)
  DM         ionsDM=ctx->ionsDM;
  RealVector *vel;
  PetscReal  vnx, vny, vnz;                           // neutral-particle velocity components
  PetscReal  vix, viy, viz;                           // ion velocity components
  PetscReal  vrx, vry, vrz;                           // ion-neutral relative-velocity components
  PetscReal  vrr;                                     // ion-neutral relative-velocity magnitude
  PetscReal  vcx, vcy, vcz;                           // center-of-mass velocity components
  PetscReal  vcr;                                     // the ion speed with respect to the center of mass
  PetscReal  costht, sintht, cosphi, sinphi;
  PetscReal  ux, uy, uz, uperp, uphi;                 // components of the unit scattering vector
  PetscReal  ux1, uy1, uz1, ux2, uy2, uz2;
  PetscReal  vfx, vfy, vfz, vfr;                      // components and magnitude of current ion's final speed
  long       seed=ctx->seed;
  PetscReal  ratio;                                   // ratio of current ion's final speed to thermal speed

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Compute the maximum relative velocity.
  vrm = 4.0*viT + PetscSqrtReal(PetscSqr(vi0x-vn0x) + PetscSqr(vi0y-vn0y) + PetscSqr(vi0z-vn0z));

  // Compute the number of collisions to attempt.
  Nc = (PetscInt)(((PetscReal)Np * fc * vrm) / ((vnT + vn0) * PetscSqrtReal(mn / mi)));
  if (Nc > Np) {
    Nc = Np;
  } else if (Nc < 0) {
    Nc = 0;
  }

  // Get an array representation of the ion velocities.
  PetscCall(DMSwarmGetField(ionsDM, "velocity", NULL, NULL, (void **)&vel));

  // Report number of collisions.
  PRINT_WORLD("Colliding %d particles out of %d ...\n", Nc, Np);

  // Attempt collisions until we reach the required number.
  while (Ns < Nc) {
    // Choose a random ion from the full distribution.
    ip = (PetscInt)(Np*ran3(&seed));

    // Store the ion velocity components.
    vix = vel[ip].x;
    viy = vel[ip].y;
    viz = vel[ip].z;

    // Choose a random neutral-particle velocity.
    vnx = gasdev(&seed)*vnT + vn0x;
    vny = gasdev(&seed)*vnT + vn0y;
    vnz = gasdev(&seed)*vnT + vn0z;

    // Compute the ion-neutral relative velocity.
    vrx = vix - vnx;
    vry = viy - vny;
    vrz = viz - vnz;
    vrr = PetscSqrtReal(PetscSqr(vrx) + PetscSqr(vry) + PetscSqr(vrz));

    // Collide only if the squared relative velocity is greater than the square
    // of a random percentage of the maximum relative velocity.
    if (PetscSqr(ran3(&seed)) < PetscSqr(vrr) / PetscSqr(vrm)) {

      // Compute the center-of-mass (CoM) velocity.
      vcx = (vix*mi + vnx*mn) / M;
      vcy = (viy*mi + vny*mn) / M;
      vcz = (viz*mi + vnz*mn) / M;

      // Compute the particle speed relative to the CoM velocity.
      vcr = PetscSqrtReal(PetscSqr(vix-vcx) + PetscSqr(viy-vcy) + PetscSqr(viz-vcz));

      // Compute the unit scattering vector relative to the CoM z axis.
      uz = 2.0*ran3(&seed) - 1.0;
      uperp = PetscSqrtReal(1.0 - PetscSqr(uz));
      uphi = 2*PETSC_PI * ran3(&seed);
      ux = uperp*PetscCosReal(uphi);
      uy = uperp*PetscSinReal(uphi);

      // Rotate the CoM frame to align its z axis with the incident direction.
      costht = vrz / vrr;
      sintht = PetscSqrtReal(1.0 - PetscSqr(costht));
      cosphi = vrx / (vrr*sintht);
      sinphi = vry / (vrr*sintht);

      /* Rotate the unit scattering vector to the incident coordinate system.
      1. rotation about CoM y axis:          (xc, yc, zc) -> (xp, yp, zp)
      2. rotation about intermediate z axis: (xp, yp, zp) -> (xi, yi, zi)
      */
      ux1 = uz*sintht + ux*costht;
      uy1 = uy;
      uz1 = uz*costht - ux*sintht;
      ux2 = ux1*cosphi - uy1*sinphi;
      uy2 = ux1*sinphi + uy1*cosphi;
      uz2 = uz1;

      /* Assign final CoM velocity components.
      vfx = vcr * ((uz*sintht + ux*costht)*cosphi - uy*sinphi)
      vfy = vcr * ((uz*sintht - ux*costht)*sinphi + uy*cosphi)
      vfz = vcr * ( uz*costht - ux*sintht                    )
      */
      vfx = vcx + vcr*ux2;
      vfy = vcy + vcr*uy2;
      vfz = vcz + vcr*uz2;

      /* Finalize
      - if result is unphysical, do not use it
      - otherwise, update the ion velocity components and count the collision
      */
      vfr = PetscSqrtReal(PetscSqr(vfx) + PetscSqr(vfy) + PetscSqr(vfz));
      ratio = vfr / viT;
      if (ratio > 10) {
        PRINT_WORLD("Warning: Refusing to accept collision that results in final speed = %4.1f times thermal speed\n", ratio);
        Nf++;
        // Terminate the simulation if at least 10 collisions have failed.
        if (Nf >= 10) {
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SIG, "Failed to collide %d ion-neutral pairs. Aborting.", Nf);
        }
      } else {
        vel[ip].x = vfx;
        vel[ip].y = vfy;
        vel[ip].z = vfz;
        Ns++;
      }

    }
  }

  // Restore the ion-velocities array.
  PetscCall(DMSwarmRestoreField(ionsDM, "velocity", NULL, NULL, (void **)&vel));

  // Report the number of actual collisions.
  PRINT_WORLD("Collision efficiency: %6.4f.\n", (PetscReal)Ns/(PetscReal)(Ns+Nf));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Update the ion velocities based on external forces and collisions. */
PetscErrorCode UpdateVelocities(KSP ksp, Context *ctx)
{
  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Apply the Boris mover to integrate dv/dt = E + vxB.
  PetscCall(BorisMover(ksp, ctx));

  // Apply the appropriate collision algorithm.
  if (ctx->neutrals.m > 0.0) {
    PetscCall(ComputeCollisions(ctx));
  }

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Update the ion positions according to $\frac{d\vec{r}}{dt} = \vec{v}$. */
PetscErrorCode UpdatePositions(Context *ctx)
{
  DM          swarm=ctx->ionsDM;
  RealVector  *pos, *vel;
  PetscInt    ip, np;
  PetscReal   x, y, z;
  PetscReal   Lx=ctx->grid.L.x;
  PetscReal   Ly=ctx->grid.L.y;
  PetscReal   Lz=ctx->grid.L.z;
  PetscReal   x0=ctx->grid.p0.x;
  PetscReal   y0=ctx->grid.p0.y;
  PetscReal   z0=ctx->grid.p0.z;
  PetscReal   x1=ctx->grid.p1.x;
  PetscReal   y1=ctx->grid.p1.y;
  PetscReal   z1=ctx->grid.p1.z;
  PetscReal   dt=ctx->dt;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Get an array representation of the ion positions.
  PetscCall(DMSwarmGetField(swarm, "position", NULL, NULL, (void **)&pos));

  // Get an array representation of the ion velocities.
  PetscCall(DMSwarmGetField(swarm, "velocity", NULL, NULL, (void **)&vel));

  // Get the number of particles on this rank.
  PetscCall(DMSwarmGetLocalSize(swarm, &np));

  // Loop over ions.
  for (ip=0; ip<np; ip++) {
    // Update the x position.
    x = pos[ip].x + vel[ip].x*dt;
    // [DEV] Apply periodic BC along x.
    while (x < x0) {
      x += Lx;
    }
    while (x > x1) {
      x -= Lx;
    }
    // Update the y position.
    y = pos[ip].y + vel[ip].y*dt;
    // [DEV] Apply periodic BC along y.
    while (y < y0) {
      y += Ly;
    }
    while (y > y1) {
      y -= Ly;
    }
    // Update the z position.
    z = pos[ip].z + vel[ip].z*dt;
    // [DEV] Apply periodic BC along z.
    while (z < z0) {
      z += Lz;
    }
    while (z > z1) {
      z -= Lz;
    }
    // TODO: Implement more flexible BC.
  }
  // Restore the ion-positions array.
  PetscCall(DMSwarmRestoreField(swarm, "position", NULL, NULL, (void **)&pos));

  // Restore the ion-velocities array.
  PetscCall(DMSwarmRestoreField(swarm, "velocity", NULL, NULL, (void **)&vel));

  // Update the swarm.
  PetscCall(DMSwarmMigrate(swarm, PETSC_TRUE));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Get the global vector (and array) corresponding to a named vlasov field. */
PetscErrorCode GetGlobalVlasovField(const char *name, Vec *vec, void *array, Context *ctx)
{
  DM        vlasovDM=ctx->vlasovDM;
  Vec       full=ctx->vlasov;
  PetscInt  nf;
  char      **names;
  DM        *dms, dm;
  PetscInt  field;
  PetscBool found;

  PetscFunctionBeginUser;

  PetscCall(DMCreateFieldDecomposition(vlasovDM, &nf, &names, NULL, &dms));
  for (field=0; field<nf; field++) {
    PetscCall(PetscStrcmp(name, names[field], &found));
    if (found) {
      dm = dms[field];
      PetscCall(DMGetGlobalVector(dm, vec));
      PetscCall(VecStrideGather(full, field, *vec, INSERT_VALUES));
      PetscCall(PetscObjectSetName((PetscObject)*vec, names[field]));
      if (array) {
        PetscCall(DMDAVecGetArray(dm, vec, (PetscReal ****)array));
      }
      break;
    }
  }

  // Release memory.
  for (field=0; field<nf; field++) {
    PetscFree(names[field]);
    PetscCall(DMDestroy(&dms[field]));
  }
  PetscFree(names);
  PetscFree(dms);

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Get the local vector (and array) corresponding to a named vlasov field. */
PetscErrorCode GetLocalVlasovField(const char *name, Vec *vec, void *array, Context *ctx)
{
  DM        vlasovDM=ctx->vlasovDM;
  Vec       full=ctx->vlasov, global;
  PetscInt  nf;
  char      **names;
  DM        *dms, dm;
  PetscInt  field;
  PetscBool found;

  PetscFunctionBeginUser;

  PetscCall(DMCreateFieldDecomposition(vlasovDM, &nf, &names, NULL, &dms));
  for (field=0; field<nf; field++) {
    PetscCall(PetscStrcmp(name, names[field], &found));
    if (found) {
      dm = dms[field];
      PetscCall(DMGetGlobalVector(dm, &global));
      PetscCall(VecStrideGather(full, field, &global, INSERT_VALUES));
      PetscCall(DMGetLocalVector(dm, &vec));
      PetscCall(DMGlobalToLocalBegin(dm, global, INSERT_VALUES, vec));
      PetscCall(DMGlobalToLocalEnd(dm, global, INSERT_VALUES, vec));
      PetscCall(PetscObjectSetName((PetscObject)vec, names[field]));
      if (array) {
        PetscCall(DMDAVecGetArray(dm, vec, (PetscReal ****)array));
      }
      break;
    }
  }

  // Release memory.
  PetscCall(DMDestroy(&vlasovDM));
  PetscCall(VecDestroy(&global));
  for (field=0; field<nf; field++) {
    PetscFree(names[field]);
    PetscCall(DMDestroy(&dms[field]));
  }
  PetscFree(names);
  PetscFree(dms);

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Restore a vector (and array) retrieved via GetGlobalVlasovField. */
PetscErrorCode RestoreGlobalVlasovField(const char *name, Vec *vec, void *array, Context *ctx)
{
  DM        vlasovDM=ctx->vlasovDM;
  Vec       full=ctx->vlasov;
  PetscInt  nf;
  char      **names;
  DM        *dms, dm;
  PetscInt  field;
  PetscBool found;

  PetscFunctionBeginUser;

  PetscCall(DMCreateFieldDecomposition(vlasovDM, &nf, &names, NULL, &dms));
  for (field=0; field<nf; field++) {
    PetscCall(PetscStrcmp(name, names[field], &found));
    if (found) {
      dm = dms[field];
      PetscCall(DMRestoreGlobalVector(dm, vec));
      if (array) {
        PetscCall(DMDAVecRestoreArray(dm, vec, (PetscReal ****)array));
      }
      break;
    }
  }

  // Release memory.
  PetscCall(DMDestroy(&vlasovDM));
  for (field=0; field<nf; field++) {
    PetscFree(names[field]);
    PetscCall(DMDestroy(&dms[field]));
  }
  PetscFree(names);
  PetscFree(dms);

  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Restore a vector (and array) retrieved via GetLocalVlasovField. */
PetscErrorCode RestoreLocalVlasovField(const char *name, Vec *vec, void *array, Context *ctx)
{
  DM        vlasovDM=ctx->vlasovDM;
  Vec       full=ctx->vlasov;
  PetscInt  nf;
  char      **names;
  DM        *dms, dm;
  PetscInt  field;
  PetscBool found;

  PetscFunctionBeginUser;

  PetscCall(DMCreateFieldDecomposition(vlasovDM, &nf, &names, NULL, &dms));
  for (field=0; field<nf; field++) {
    PetscCall(PetscStrcmp(name, names[field], &found));
    if (found) {
      dm = dms[field];
      if (array) {
        PetscCall(DMDAVecRestoreArray(dm, vec, (PetscReal ****)array));
      }
      PetscCall(DMRestoreLocalVector(dm, vec));
      break;
    }
  }

  // Release memory.
  PetscCall(DMDestroy(&vlasovDM));
  for (field=0; field<nf; field++) {
    PetscFree(names[field]);
    PetscCall(DMDestroy(&dms[field]));
  }
  PetscFree(names);
  PetscFree(dms);

  PetscFunctionReturn(PETSC_SUCCESS);
}

