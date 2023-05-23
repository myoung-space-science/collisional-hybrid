#ifndef HYBRID_H
#define HYBRID_H

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
#define PRINTLINE {PRINT_WORLD("%d\n", __LINE__);}

#define NDIM 3
#define Q 1.6022e-19   // fundamental charge in C
#define MP 1.6726e-27  // proton mass in kg
#define ME 9.1094e-31  // electron mass in kg
#define KB  1.3807e-23 // Boltzmann constant

// Temporary declaration of number of particles per cell, per dimension.
#define NPPCELL 1
// Temporary declaration of total number of particles per distribution.
#define NPTOTAL 32768

// The number of non-null values in the LHS-operator matrix stencil. This is
// equivalent to the standard box stencil with the corners removed.
#define NVALUES 19

typedef struct {
  PetscInt x; // x component
  PetscInt y; // y component
  PetscInt z; // z component
} IntVector;

typedef struct {
  PetscReal x; // x component
  PetscReal y; // y component
  PetscReal z; // z component
  PetscReal r; // magnitude
} RealVector;

typedef struct {
  IntVector  N;  // number of cells
  RealVector L;  // physical length
  RealVector d;  // physical cell spacing
  RealVector p0; // lower physical bound
  RealVector p1; // upper physical bound
} Grid;

typedef struct {
  PetscReal  q;     // charge
  PetscReal  m;     // mass
  PetscReal  nu;    // frequency of collisions with neutral particles
  RealVector Omega; // gyrofrequency components
  RealVector kappa; // magnetization components
  RealVector v0;    // drift velocity
  RealVector vT;    // thermal velocity
  PetscReal  gamma; // thermal coefficient
  PetscReal  T;     // temperature
} Charged;

typedef struct {
  PetscReal  m;  // mass
  RealVector v0; // drift velocity
  PetscReal  vT; // thermal velocity
  PetscReal  T;  // temperature
} Neutral;

typedef struct {
  RealVector B0; // constant magnetic-field amplitude
  RealVector E0; // constant electric-field amplitude
  PetscInt   Np; // number of charged particles
} Plasma;

typedef struct {
  PetscMPIInt rank; // global processor number
  PetscMPIInt size; // total number of processors
} MPIContext;

extern const char *RHSTypes[];

typedef enum {
  RHS_CONSTANT,
  RHS_SINUSOIDAL,
  RHS_FULL,
} RHSType;

extern const char *LHSTypes[];

typedef enum {
  LHS_IDENTITY,
  LHS_LAPLACIAN,
  LHS_FULL,
} LHSType;

extern const char *DensityTypes[];

typedef enum {
  DENSITY_FLAT_SOBOL,
  DENSITY_FLAT_REVERSE,
  DENSITY_FLAT_NORMAL,
  DENSITY_UNIFORM,
  DENSITY_SINUSOIDAL,
  DENSITY_GAUSSIAN,
} DensityType;

extern const char *BCTypes[];

typedef enum {
  BC_PERIODIC,
  BC_INJECTION,
  BC_REFLECTION,
  BC_DIRICHLET,
  BC_NEUMANN,
} BCType;

typedef PetscErrorCode (*StencilFunc)(PetscInt i, PetscInt j, PetscInt k, PetscReal ***f, MatStencil cols[NVALUES], PetscReal vals[NVALUES], void *ctx);
typedef PetscErrorCode (*LHSFunc)(KSP ksp, Mat J, Mat A, void *_ctx);
typedef PetscErrorCode (*RHSFunc)(KSP ksp, Vec b, void *_ctx);

typedef struct {
  Grid           grid;         // grid information
  Charged        electrons;    // electron parameter values
  Charged        ions;         // ion parameter values
  Neutral        neutrals;     // neutral-particle parameter values
  Plasma         plasma;       // plasma information
  MPIContext     mpi;          // MPI information
  Vec            vlasov;       // full vector of all Vlasov quantities
  DM             swarm;        // PIC-swarm data manager
  Vec            phi;          // electrostatic potential
  Vec            rhs;          // potential-equation forcing vector
  PetscBool      viewLHS;      // option to view LHS operator structure
  BCType         xBC[2];       // x-axis boundary condition(s)
  BCType         yBC[2];       // y-axis boundary condition(s)
  BCType         zBC[2];       // z-axis boundary condition(s)
  DMBoundaryType xDMBC;        // x-axis boundary type for the DM
  DMBoundaryType yDMBC;        // y-axis boundary type for the DM
  DMBoundaryType zDMBC;        // z-axis boundary type for the DM
  RHSType        rhsType;      // type of RHS vector to use
  LHSType        lhsType;      // type of LHS operator to use
  RHSFunc        rhsFunc;      // function corresponding to rhsType
  LHSFunc        lhsFunc;      // function corresponding to lhsType
  StencilFunc    stencilFunc;  // LHS stencil function corresponding to boundary conditions
  DensityType    densityType;  // type of initial density profile to use
  PetscInt       Nt;           // number of time steps
  PetscReal      dt;           // time-step width
  PetscInt       it;           // time-step counter
  long           seed;         // random-number seed
} Context;

typedef struct {
  PetscScalar n;          // scalar density
  PetscScalar flux[NDIM]; // vector flux
} GridNode;

#endif // HYBRID_H