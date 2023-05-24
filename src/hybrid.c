/* 3D Hybrid PIC */
static char help[] = "A 3D hybrid particle-in-cell (PIC) simulation.";

#include <time.h>
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include "hybrid.h"
#include "parameters.h"
#include "setup.h"
#include "random.h"
#include "distributions.h"
#include "particles.h"
#include "potential.h"
#include "lhs.h"
#include "rhs.h"
#include "vectors.h"
#include "output.h"


static PetscErrorCode
BorisMover(KSP ksp, Context *ctx)
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
static PetscErrorCode
ComputeCollisions(Context *ctx)
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


static PetscErrorCode
UpdateVelocities(KSP ksp, Context *ctx)
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
static PetscErrorCode
UpdatePositions(Context *ctx)
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


int main(int argc, char **args)
{
  MPIContext  mpi;
  time_t      startTime, endTime;
  Context     ctx;
  DM          vdm, pdm;
  KSP         ksp;
  PetscInt    it;
  char        itfmt[5];
  char        pathfmt[PETSC_MAX_PATH_LEN]="", pathstr[PETSC_MAX_PATH_LEN];
  char        stepfmt[256]="", stepstr[256];
  PetscInt    itwidth;

  PetscFunctionBeginUser;

  /* Initialize PETSc and MPI. */
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &mpi.rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi.size));
  if (mpi.rank == 0) {
    time(&startTime);
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n**************** START *****************\n\n"));

  /* Assign parameter values from user arguments or defaults. */
  PetscCall(ProcessOptions(&ctx));

  /* Store MPI information in the application context. */
  ctx.mpi = mpi;

  /* Define a rank-specific random-number seed. */
  ctx.seed = (long)(-(mpi.rank + 1)*12345);

  /* Set up the discrete grid of Vlasov quantities. */
  PetscCall(SetUpVlasovDM(&vdm, &ctx));

  /* Create a persistent vector for outputing Vlasov quantities. */
  PetscCall(DMCreateGlobalVector(vdm, &ctx.vlasov));
  PetscCall(VecZeroEntries(ctx.vlasov));

  /* Set up the particle swarm for ions. */
  PetscCall(SetUpIonsDM(vdm, &ctx));

  /* Set initial particle positions. */
  PetscCall(InitializePositions(&ctx));

  /* Set initial particle velocities. */
  PetscCall(InitializeVelocities(&ctx));

  /* Echo the initial state. */
  if (mpi.rank == 0) {
    PetscCall(EchoOptions(ctx));
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  /* Compute initial density and flux. */
  PetscCall(CollectVlasovQuantities(&ctx));

  /* Set up the discrete grid for the electrostatic potential. */
  PetscCall(SetUpPotentialDM(&pdm, &ctx));

  /* Set up the Krylov-solver context for the electrostatic potential. */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, pdm));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetComputeInitialGuess(ksp, ComputeInitialPhi, &ctx));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, &ctx));
  PetscCall(KSPSetComputeOperators(ksp, ComputeLHS, &ctx));

  /* Compute initial electrostatic potential. */
  PetscCall(ComputePotential(ksp, &ctx));

  /* Create a format string for the time step. */
  itwidth = 1+PetscLog10Real(ctx.Nt);
  sprintf(itfmt, "%%0%dd", itwidth);

  /* Create a template for time-dependent filenames. */
  PetscCall(PetscStrcat(pathfmt, "arrays-"));
  PetscCall(PetscStrcat(pathfmt, itfmt));
  PetscCall(PetscStrcat(pathfmt, ".hdf"));

  /* Output initial conditions. */
  sprintf(pathstr, "arrays-initial.hdf");
  PetscCall(OutputHDF5(pathstr, &ctx));

  /* Create a template for the time-step string. */
  PetscCall(PetscStrcat(stepfmt, "Time step "));
  PetscCall(PetscStrcat(stepfmt, itfmt));
  PetscCall(PetscStrcat(stepfmt, "\n"));

  PRINT_WORLD("\n*** Main time-step loop ***\n\n");
  /* Begin main time-step loop. */
  for (it=0; it<ctx.Nt; it++) {

    /* Create a string to display time step with the appropriate width. */
    sprintf(stepstr, stepfmt, it);
    PRINT_WORLD(stepstr);

    /* Update velocities */
    PetscCall(UpdateVelocities(ksp, &ctx));

    /* Update positions. */
    PetscCall(UpdatePositions(&ctx));

    /* Compute density and flux from ion positions. */
    PetscCall(CollectVlasovQuantities(&ctx));

    /* Compute potential from density. */
    PetscCall(ComputePotential(ksp, &ctx));

    /* Output current time step. */
    sprintf(pathstr, pathfmt, it);
    PetscCall(OutputHDF5(pathstr, &ctx));

  }

  /* Free memory. */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&ctx.vlasov));
  PetscCall(DMDestroy(&vdm));
  PetscCall(DMDestroy(&ctx.ionsDM));
  PetscCall(DMDestroy(&pdm));

  /* Write time information. */
  if (mpi.rank == 0) {
    time(&endTime);
  }
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n----------------------------------------\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Start time: %s", asctime(localtime(&startTime))));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "End time:   %s", asctime(localtime(&endTime))));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "----------------------------------------\n"));

  /* Finalize PETSc and MPI. */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n***************** END ******************\n"));
  PetscCall(PetscFinalize());

  return 0;
}