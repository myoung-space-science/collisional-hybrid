#ifndef PARTICLES_H
#define PARTICLES_H

#include "hybrid.h"

extern PetscErrorCode InitializePositions(Context *ctx);

extern PetscErrorCode InitializeVelocities(Context *ctx);

extern PetscErrorCode CollectVlasovQuantities(Context *ctx);

#endif // PARTICLES_H