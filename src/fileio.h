#ifndef OUTPUT_H
#define OUTPUT_H

#include <petsc.h>
#include "hybrid.h"

extern PetscErrorCode LoadVlasov(DM gridDM, const char *name, Vec *vlasov);

extern PetscErrorCode OutputHDF5(const char *name, Context *ctx);

extern PetscErrorCode ViewReducedLHS(PetscInt Nx, PetscInt Ny, PetscInt Nz, void *ctx);

#endif // OUTPUT_H