#ifndef VECTORS_H
#define VECTORS_H

#include <petsc.h>
#include "hybrid.h"

extern PetscErrorCode DifferenceVector(PetscReal ***F, PetscReal x0, PetscReal y0, PetscReal z0, Grid grid, PetscReal f[NDIM]);

extern PetscErrorCode DotProduct(PetscReal a[NDIM], PetscReal b[NDIM], PetscReal *c);

extern PetscErrorCode CrossProduct(PetscReal a[NDIM], PetscReal b[NDIM], PetscReal c[NDIM]);

#endif // VECTORS_H