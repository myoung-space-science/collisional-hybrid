#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include <petsc.h>
#include "hybrid.h"

// Type to be used for particle distribution functions.
typedef PetscErrorCode (*DistributionFunction)(PetscReal x, PetscReal y, PetscReal z, PetscReal *v, Context *ctx);

extern PetscErrorCode UniformDistribution_FromSwarm(Context *ctx);

extern PetscErrorCode UniformDistribution(Context *ctx);

extern PetscErrorCode SobolDistribution(Context *ctx);

extern PetscErrorCode SinusoidalDistribution(PetscReal x, PetscReal y, PetscReal z, PetscReal *v, Context *ctx);

extern PetscErrorCode Rejection(DistributionFunction density, Context *ctx);

#endif // DISTRIBUTIONS_H