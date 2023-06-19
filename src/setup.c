#include <petsc.h>
#include "hybrid.h"


/* Set parameter values for the vlasov-quantities data manager. */
PetscErrorCode SetUpVlasovDM(Context *ctx)
{
  DM             dm;
  PetscInt       Nx=(ctx->grid.N.x > 0 ? ctx->grid.N.x : 7);
  PetscInt       Ny=(ctx->grid.N.y > 0 ? ctx->grid.N.y : 7);
  PetscInt       Nz=(ctx->grid.N.z > 0 ? ctx->grid.N.z : 7);
  DMBoundaryType xBC=ctx->xDMBC;
  DMBoundaryType yBC=ctx->yDMBC;
  DMBoundaryType zBC=ctx->zDMBC;
  PetscInt       dof=4;
  PetscInt       width=1;

  PetscFunctionBeginUser;
  ECHO_FUNCTION_ENTER;

  // Create the DM.
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, xBC, yBC, zBC, DMDA_STENCIL_BOX, Nx, Ny, Nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, width, NULL, NULL, NULL, &dm));
  // Perform basic setup.
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm, "vlasov_"));
  PetscCall(DMDASetElementType(dm, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(PetscObjectSetName((PetscObject)(dm), "Vlasov"));
  // Synchronize values of Nx, Ny, and Nz passed via -n{x,y,z} or
  // -da_grid_{x,y,z}. Note that this gives precedence to the latter.
  PetscCall(DMDAGetInfo(dm, NULL, &Nx, &Ny, &Nz, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
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
  ctx->grid.d.x = ctx->grid.L.x / (PetscReal)ctx->grid.N.x;
  ctx->grid.d.y = ctx->grid.L.y / (PetscReal)ctx->grid.N.y;
  ctx->grid.d.z = ctx->grid.L.z / (PetscReal)ctx->grid.N.z;
  // Set uniform coordinates on the DM.
  PetscCall(DMDASetUniformCoordinates(dm, ctx->grid.p0.x, ctx->grid.p1.x, ctx->grid.p0.y, ctx->grid.p1.y, ctx->grid.p0.z, ctx->grid.p1.z));
  // Declare grid-quantity names.
  PetscCall(DMDASetFieldName(dm, 0, "density"));
  PetscCall(DMDASetFieldName(dm, 1, "x flux"));
  PetscCall(DMDASetFieldName(dm, 2, "y flux"));
  PetscCall(DMDASetFieldName(dm, 3, "z flux"));
  // View information about the DM.
  PetscCall(DMView(dm, PETSC_VIEWER_STDOUT_WORLD));
  // Create a persistent vector for outputing Vlasov quantities.
  PetscCall(DMCreateGlobalVector(dm, &ctx->vlasov));
  PetscCall(VecZeroEntries(ctx->vlasov));
  // Assign the vlasov DM to the application context.
  ctx->vlasovDM = dm;

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Set parameter values for the electrostatic-potential data manager. */
PetscErrorCode SetUpPotentialDM(DM *dm, Context *ctx)
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
  // Associate the user context with this DM.
  PetscCall(DMSetApplicationContext(*dm, &ctx));
  // Echo information about the DM.
  PetscCall(DMView(*dm, PETSC_VIEWER_STDOUT_WORLD));

  ECHO_FUNCTION_EXIT;
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Set parameter values for the ion-swarm data manager. */
PetscErrorCode SetUpIonsDM(Context *ctx)
{
  DM       ionsDM, vlasovDM=ctx->vlasovDM;
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


