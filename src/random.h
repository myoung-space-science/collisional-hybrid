#include <petsc.h>

#define IA 16807
#define IM 2147483647
#define AM (1.0 / IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1 + (IM-1) / NTAB)
#define EPS PETSC_MACHINE_EPSILON
#define RNMX (1.0 - EPS)

#define MBIG 1000000000
#define MSEED 161803398
#define MZ 0
#define FAC (1.0 / MBIG)


/* Adaptation of ran1 from Numerical Recipes, 2nd edition. */
float ran1(long *idum)
{
  int j;
  long k;
  static long iy=0;
  static long iv[NTAB];
  float temp;

  /* Initialize the random-number generator. */
  if (*idum <= 0 || !iy) {
    // Be sure to prevent idum == 0.
    if (-(*idum) < 1) {
      *idum = 1;
    } else {
      *idum = -(*idum);
    }
    // Load the shuffle table, after 8 warm-ups.
    for (j=NTAB+7; j>=0; j--) {
      k = (*idum) / IQ;
      *idum = IA * (*idum - k*IQ) - IR*k;
      if (*idum < 0) {
        *idum += IM;
      }
      if (j < NTAB) {
        iv[j] = *idum;
      }
    }
    iy = iv[0];
  }
  /* Start here after initialization. */
  k = (*idum) / IQ;
  // Compute idum=(IA*idum) % IM without overflows by Schrage's method.
  *idum = IA * (*idum - k*IQ) - IR*k;
  if (*idum < 0) {
    *idum += IM;
  }
  // Set j, which will be in the range 0..NTAB-1
  j = iy / NDIV;
  // Output the previously stored value and refill the shuffle table. If the
  // result is larger than the predefined RNMX value, this will output RNMX
  // because users don't expect an endpoint value.
  iy = iv[j];
  iv[j] = *idum;
  if ((temp = AM*iy) > RNMX) {
    return RNMX;
  } else {
    return temp;
  }
}


/* Wrapper for local implementation of ran1.

This function stores the result of ran1 in a user-provided variable, in order to
leverage the PETSc error-checking machinery. Users may also directly use ran1 in
functional form.
*/
static PetscErrorCode
Ran1(long *idum, PetscReal *result)
{
  PetscFunctionBeginUser;
  *result = (PetscReal)ran1(idum);
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Adaptation of ran3 from Numerical Recipes, 2nd edition.

Comments appear close to their position in the original text. In addition, they
note that the 56 in ma[56] and the 31 in inextp = 31 are special values.
*/
float ran3(long *idum)
{
  static int  inext, inextp;
  static long ma[56];
  static int  iff=0;
  long        mj, mk;
  int         i, ii, k;

  /* Initialize the random-number generator. */
  if (*idum < 0 || iff == 0) {
    iff = 1;
    // Initialize ma[55] using the seed (idum) and the large number MSEED.
    mj = labs(MSEED - labs(*idum));
    mj %= MBIG;
    ma[55] = mj;
    mk = 1;
    // Initialize the rest of the table, in a slightly random order, with
    // numbers that are not especially random.
    for (i=1; i<=54; i++) {
      ii = (21*i) % 55;
      ma[ii] = mk;
      mk = mj-mk;
      if (mk < MZ) {
        mk += MBIG;
      }
      mj = ma[ii];
      // Randomize them by "warming up the generator".
      for (k=1; k<=4; k++) {
        for (i=1; i<=55; i++) {
          ma[i] -= ma[1+(i+30) % 55];
          if (ma[i] < MZ) {
            ma[i] += MBIG;
          }
        }
      }
      // Prepare indices for the first generated number.
      inext = 0;
      inextp = 31;
      *idum = 1;
    }
    /* Start here after initialization. */
    // Increment inext, wrapping around 56 to 1.
    if (++inext == 56) {
      inext = 1;
    }
    // Increment inextp, wrapping around 56 to 1.
    if (++inextp == 56) {
      inextp = 1;
    }
    // Generate a new random number subtractively.
    mj = ma[inext]-ma[inextp];
    // Make sure it is in range.
    if (mj < MZ) {
      mj += MBIG;
    }
    // Store the new number.
    ma[inext] = mj;
    // Output the derived uniform deviate.
    return mj*FAC;
  }
}


/* Wrapper for local implementation of ran3.

This function stores the result of ran3 in a user-provided variable, in order to
leverage the PETSc error-checking machinery. Users may also directly use ran3 in
functional form.
*/
static PetscErrorCode
Ran3(long *idum, PetscReal *result)
{
  PetscFunctionBeginUser;
  *result = (PetscReal)ran3(idum);
  PetscFunctionReturn(PETSC_SUCCESS);
}


