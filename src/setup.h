#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <petsc.h>
#include "hybrid.h"

extern PetscErrorCode SetUpVlasovDM(Context *ctx);

extern PetscErrorCode SetUpPotentialDM(DM *dm, Context *ctx);

extern PetscErrorCode SetUpIonsDM(Context *ctx);

#endif // INITIALIZE_H