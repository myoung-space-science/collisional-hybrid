#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <petsc.h>
#include "hybrid.h"

extern PetscErrorCode SetUpVlasovDM(DM *dm, Context *ctx);

extern PetscErrorCode SetUpPotentialDM(DM *dm, Context *ctx);

extern PetscErrorCode SetUpIonsDM(DM vlasovDM, Context *ctx);

#endif // INITIALIZE_H