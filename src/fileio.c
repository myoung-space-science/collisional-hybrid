#include <petsc.h>
#include <petscviewerhdf5.h>
#include "hybrid.h"


PetscErrorCode LoadVlasovQuantities(Context *ctx)
{
  PetscBool   nullPath;
  PetscViewer viewer;
  DM          *dms, dm, vlasovDM=ctx->vlasovDM;
  PetscInt    Nf;
  char        **keys;
  PetscInt    field;
  Vec         current, vlasov=ctx->vlasov;
  char        key[2048];

  PetscFunctionBeginUser;

  // Raise an error if the user did not provide an input file.
  PetscCall(PetscStrcmp(ctx->inpath, "", &nullPath));
  if (nullPath) {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Missing input file.");
  }

  // Create the HDF5 viewer.
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, ctx->inpath, FILE_MODE_READ, &viewer));

  // Zero the target vlasov vector.
  PetscCall(VecZeroEntries(vlasov));

  // Load vlasov quantities from the HDF5 file.
  PRINT_WORLD("Attempting to load arrays from %s\n", ctx->inpath);
  PetscCall(DMCreateFieldDecomposition(vlasovDM, &Nf, &keys, NULL, &dms));
  for (field=0; field<Nf; field++) {
    dm = dms[field];
    PetscCall(DMGetGlobalVector(dm, &current));
    sprintf(key, "arrays-kji/%s", keys[field]);
    PetscCall(PetscObjectSetName((PetscObject)current, key));
    PetscCall(VecLoad(current, viewer));
    PetscCall(VecStrideScatter(current, field, vlasov, INSERT_VALUES));
    PRINT_WORLD("Loaded \"%s\"\n", key);
    PetscCall(DMRestoreGlobalVector(dm, &current));
  }

  // Release memory.
  for (field=0; field<Nf; field++) {
    PetscFree(keys[field]);
    PetscCall(DMDestroy(&dms[field]));
  }
  PetscFree(keys);
  PetscFree(dms);

  // Destroy the HDF5 viewer.
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode OutputHDF5(const char *insert, Context *ctx)
{
  char        name[PETSC_MAX_PATH_LEN]="";
  PetscViewer viewer;
  DM          vlasovDM=ctx->vlasovDM, *dms, dm;
  PetscInt    Nf;
  char        **keys;
  PetscInt    field;
  Vec         target, vlasov=ctx->vlasov, rhs=ctx->rhs, phi=ctx->phi;

  PetscFunctionBeginUser;

  // Build the full file name.
  PetscCall(PetscStrcat(name, ctx->outstem));
  PetscCall(PetscStrcat(name, insert));
  PetscCall(PetscStrcat(name, ".hdf"));

  // Create the HDF5 viewer.
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, name, FILE_MODE_WRITE, &viewer));

  // Write vlasov quantities to the HDF5 file.
  PetscCall(DMCreateFieldDecomposition(vlasovDM, &Nf, &keys, NULL, &dms));
  for (field=0; field<Nf; field++) {
    dm = dms[field];
    PetscCall(DMGetGlobalVector(dm, &target));
    PetscCall(VecStrideGather(vlasov, field, target, INSERT_VALUES));
    PetscCall(PetscObjectSetName((PetscObject)target, keys[field]));
    PetscCall(VecView(target, viewer));
    PetscCall(DMRestoreGlobalVector(dm, &target));
  }

  // Release memory.
  for (field=0; field<Nf; field++) {
    PetscFree(keys[field]);
    PetscCall(DMDestroy(&dms[field]));
  }
  PetscFree(keys);
  PetscFree(dms);

  // Write the forcing vector to the HDF5 file.
  PetscCall(PetscObjectSetName((PetscObject)rhs, "rhs"));
  PetscCall(VecView(rhs, viewer));

  // Write the electrostatic potential to the HDF5 file.
  PetscCall(PetscObjectSetName((PetscObject)phi, "potential"));
  PetscCall(VecView(phi, viewer));

  // Destroy the HDF5 viewer.
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(PETSC_SUCCESS);
}


/*
Notes
-----
* This function is supposed to provide a way to view the basic structure of the
  LHS operator matrix, even for production-sized runs, without potentially
  creating a very large binary file.
* It doesn't currently work because the local N{x,y,z} arguments are not
  necessarily consistent with their corresponding values in the application
  context. We could consider defining a new context to pass to this function,
  but that may require refactoring ComputeLHS.
*/
PetscErrorCode ViewReducedLHS(PetscInt Nx, PetscInt Ny, PetscInt Nz, void *_ctx)
{
  Context        *ctx=(Context *)_ctx;
  KSP            ksp;
  DM             dm;
  DMBoundaryType xBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType yBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType zBC=DM_BOUNDARY_PERIODIC;
  Mat            A;
  PetscViewer    viewer;

  PetscFunctionBeginUser;

  PetscCall(DMDACreate3d(
            PETSC_COMM_WORLD,
            xBC, yBC, zBC,
            DMDA_STENCIL_BOX,
            Nx, Ny, Nz,
            PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
            1, 1,
            NULL, NULL, NULL,
            &dm));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, dm));
  PetscCall(KSPSetComputeOperators(ksp, ctx->lhsFunc, &ctx));
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "lhs.dat", FILE_MODE_WRITE, &viewer));
  PetscCall(MatView(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&dm));
  PetscCall(KSPDestroy(&ksp));

  PetscFunctionReturn(PETSC_SUCCESS);
}


