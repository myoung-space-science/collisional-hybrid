#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <petsc.h>
#include "hybrid.h"

extern PetscErrorCode InitializeVlasovDM(DM *dm, Context *ctx);

extern PetscErrorCode InitializePotentialDM(DM *dm, Context *ctx);

extern PetscErrorCode InitializeIonsDM(DM vlasovDM, Context *ctx);

extern PetscErrorCode InitializePositions(Context *ctx);

extern PetscErrorCode InitializeVelocities(Context *ctx);

#endif // INITIALIZE_H