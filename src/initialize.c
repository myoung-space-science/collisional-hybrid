#include <petsc.h>
#include "hybrid.h"
#include "distributions.h"

/* Set parameter values for the vlasov-quantities data manager. */
PetscErrorCode InitializeVlasovDM(DM *dm, Context *ctx)
{
  PetscInt       Nx=(ctx->grid.N.x > 0 ? ctx->grid.N.x : 7);
  PetscInt       Ny=(ctx->grid.N.y > 0 ? ctx->grid.N.y : 7);
  PetscInt       Nz=(ctx->grid.N.z > 0 ? ctx->grid.N.z : 7);
  DMBoundaryType xBC=DM_BOUNDARY_GHOSTED;
  DMBoundaryType yBC=DM_BOUNDARY_GHOSTED;
  DMBoundaryType zBC=DM_BOUNDARY_GHOSTED;
  PetscReal      dx, dy, dz;
  PetscInt       dof=4;
  PetscInt       width=1;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Create the DM.
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, xBC, yBC, zBC, DMDA_STENCIL_BOX, Nx, Ny, Nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, width, NULL, NULL, NULL, dm));
  // Perform basic setup.
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*dm), "vlasov_"));
  PetscCall(DMDASetElementType(*dm, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetUp(*dm));
  PetscCall(PetscObjectSetName((PetscObject)(*dm), "Vlasov"));
  // Synchronize values of Nx, Ny, and Nz passed via -n{x,y,z} or
  // -da_grid_{x,y,z}. Note that this gives precedence to the latter.
  PetscCall(DMDAGetInfo(*dm, NULL, &Nx, &Ny, &Nz, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  if (ctx->grid.N.x == -1) {
    ctx->grid.N.x = Nx;
  }
  if (ctx->grid.N.y == -1) {
    ctx->grid.N.y = Ny;
  }
  if (ctx->grid.N.z == -1) {
    ctx->grid.N.z = Nz;
  }
  // Set the number of charged particles equal to the number of grid cells, if
  // necessary. Note that this must occur after synchronizing grid cells.
  if (ctx->plasma.Np == -1) {
    ctx->plasma.Np = ctx->grid.N.x * ctx->grid.N.y * ctx->grid.N.z;
  }
  // Define the physical grid-cell spacing.
  dx = ctx->grid.L.x / (PetscReal)ctx->grid.N.x;
  dy = ctx->grid.L.y / (PetscReal)ctx->grid.N.y;
  dz = ctx->grid.L.z / (PetscReal)ctx->grid.N.z;
  ctx->grid.d.x = dx;
  ctx->grid.d.y = dy;
  ctx->grid.d.z = dz;
  // Set uniform coordinates on the DM.
  PetscCall(DMDASetUniformCoordinates(*dm, ctx->grid.p0.x, ctx->grid.p1.x+dx, ctx->grid.p0.y, ctx->grid.p1.y+dy, ctx->grid.p0.z, ctx->grid.p1.z+dz));
  // Declare grid-quantity names.
  PetscCall(DMDASetFieldName(*dm, 0, "density"));
  PetscCall(DMDASetFieldName(*dm, 1, "x flux"));
  PetscCall(DMDASetFieldName(*dm, 2, "y flux"));
  PetscCall(DMDASetFieldName(*dm, 3, "z flux"));
  // Associate the user context with the DM.
  PetscCall(DMSetApplicationContext(*dm, &ctx));
  // View information about the DM.
  PetscCall(DMView(*dm, PETSC_VIEWER_STDOUT_WORLD));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Set parameter values for the electrostatic-potential data manager. */
PetscErrorCode InitializePotentialDM(DM *dm, Context *ctx)
{
  PetscInt       Nx=ctx->grid.N.x;
  PetscInt       Ny=ctx->grid.N.y;
  PetscInt       Nz=ctx->grid.N.z;
  DMBoundaryType xBC=ctx->xDMBC;
  DMBoundaryType yBC=ctx->yDMBC;
  DMBoundaryType zBC=ctx->zDMBC;
  PetscInt       dof=1;
  PetscInt       width=1;
  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Create the DM object.
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, xBC, yBC, zBC, DMDA_STENCIL_BOX, Nx, Ny, Nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, width, NULL, NULL, NULL, dm));
  // Perform basic setup.
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*dm), "potential_"));
  PetscCall(DMDASetElementType(*dm, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetUp(*dm));
  PetscCall(PetscObjectSetName((PetscObject)(*dm), "Potential"));
  // Assign the field name.
  PetscCall(DMDASetFieldName(*dm, 0, "potential"));
  // Echo information about the DM.
  PetscCall(DMView(*dm, PETSC_VIEWER_STDOUT_WORLD));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Set parameter values for the ion-swarm data manager. */
PetscErrorCode InitializeIonsDM(DM vlasovDM, Context *ctx)
{
  DM       ionsDM;
  PetscInt dim;
  PetscInt bufsize=0;
  PetscInt Np, np;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Create the ions DM.
  PetscCall(DMCreate(PETSC_COMM_WORLD, &ionsDM));
  // Perform basic setup.
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)ionsDM, "ions_"));
  PetscCall(DMSetType(ionsDM, DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject)ionsDM, "Ions"));
  // Synchronize the ions DM with the vlasov DM.
  PetscCall(DMGetDimension(vlasovDM, &dim));
  PetscCall(DMSetDimension(ionsDM, dim));
  PetscCall(DMSwarmSetCellDM(ionsDM, vlasovDM));
  // Declare this to be a PIC swarm. This must occur after setting `dim`.
  PetscCall(DMSwarmSetType(ionsDM, DMSWARM_PIC));
  // Register fields that each particle will have.
  PetscCall(DMSwarmInitializeFieldRegister(ionsDM));
  // --> (x, y, z) position components
  PetscCall(DMSwarmRegisterUserStructField(ionsDM, "position", sizeof(RealVector)));
  // --> (x, y, z) velocity components
  PetscCall(DMSwarmRegisterUserStructField(ionsDM, "velocity", sizeof(RealVector)));
  PetscCall(DMSwarmFinalizeFieldRegister(ionsDM));
  // Set the per-processor swarm size and buffer length for efficient resizing.
  Np = (ctx->plasma.Np > 0) ? ctx->plasma.Np : NPTOTAL;
  np = (PetscInt)(Np / ctx->mpi.size);
  bufsize = (PetscInt)(0.25 * np);
  PetscCall(DMSwarmSetLocalSizes(ionsDM, np, bufsize));
  // View information about the ions DM.
  PetscCall(DMView(ionsDM, PETSC_VIEWER_STDOUT_WORLD));
  // Assign the ions DM to the application context.
  ctx->ionsDM = ionsDM;

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


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


