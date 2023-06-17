#ifndef OUTPUT_H
#define OUTPUT_H

#include <petsc.h>
#include "hybrid.h"

extern PetscErrorCode LoadVlasovQuantities(Context *ctx);

extern PetscErrorCode OutputHDF5(const char *insert, Context *ctx);

extern PetscErrorCode ViewReducedLHS(PetscInt Nx, PetscInt Ny, PetscInt Nz, void *ctx);

#endif // OUTPUT_H