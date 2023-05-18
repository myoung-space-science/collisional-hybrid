#ifndef _HYBRID_H
#define _HYBRID_H

#include <petsc.h>

#define ECHO_FUNCTION_ENTER {PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n--> Entering %s(...) <--\n\n", __func__));}
#define ECHO_FUNCTION_EXIT {PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n--> Exiting %s(...) <--\n\n", __func__));}

// [DEV] Not in use. These will create a lot of output if used in a function
// that gets called often (e.g., an initial density function or random-number
// generator).
#define FUNCTION_START { \
  PetscFunctionBeginUser; \
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n--> Entering %s(...) <--\n\n", __func__)); \
}
#define FUNCTION_END { \
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n--> Exiting %s(...) <--\n\n", __func__)); \
  PetscFunctionReturn(PETSC_SUCCESS); \
}

#define PRINT_WORLD(...) {PetscCall(PetscPrintf(PETSC_COMM_WORLD, __VA_ARGS__));}
#define PRINT_RANKS(...) {PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, __VA_ARGS__)); PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));}
#define NEWLINE {PRINT_WORLD("\n");}

#endif // _HYBRID_H