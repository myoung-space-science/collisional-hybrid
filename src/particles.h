#ifndef PARTICLES_H
#define PARTICLES_H

#include <petsc.h>
#include "hybrid.h"

extern PetscErrorCode InitializePositions(Context *ctx);

extern PetscErrorCode InitializeVelocities(Context *ctx);

extern PetscErrorCode CollectVlasovQuantities(Context *ctx);

extern PetscErrorCode ApplyVlasovBC(Context *ctx);

extern PetscErrorCode BorisMover(KSP ksp, Context *ctx);

extern PetscErrorCode ComputeCollisions(Context *ctx);

extern PetscErrorCode UpdateVelocities(KSP ksp, Context *ctx);

extern PetscErrorCode UpdatePositions(Context *ctx);

extern PetscErrorCode GetGlobalVlasovField(const char *name, Vec *vec, void *array, Context *ctx);

extern PetscErrorCode RestoreGlobalVlasovField(const char *name, Vec *vec, void *array, Context *ctx);

extern PetscErrorCode GetLocalVlasovField(const char *name, Vec *vec, void *array, Context *ctx);

extern PetscErrorCode RestoreLocalVlasovField(const char *name, Vec *vec, void *array, Context *ctx);

#endif // PARTICLES_H