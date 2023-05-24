#ifndef CONSTANTS_H
#define CONSTANTS_H

#define NDIM 3
#define Q 1.6022e-19   // fundamental charge in C
#define MP 1.6726e-27  // proton mass in kg
#define ME 9.1094e-31  // electron mass in kg
#define KB  1.3807e-23 // Boltzmann constant

// The number of non-null values in the LHS-operator matrix stencil. This is
// equivalent to the standard box stencil with the corners removed.
#define NVALUES 19

#endif // CONSTANTS_H