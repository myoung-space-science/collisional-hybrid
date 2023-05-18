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

#define MAXBIT 30
#define MAXDIM 6


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


/* Adaptation of gasdev from Numerical Recipes, 2nd edition. */
float gasdev(long *idum)
{
  static int   iset=0;
  static float gset;
  PetscReal    r1, r2;
  float        fac, rsq, v1, v2;

  /* Reinitialize the sequence. */
  if (*idum < 0) {
    iset = 0;
  }
  // If we don't have an extra deviate handy, pick two uniform numbers in the
  // square extending from -1 to +1 in each direction; keep picking until they
  // fall within the unit circle.
  if (iset == 0) {
    do {
      v1 = 2.0*ran1(idum) - 1.0;
      v2 = 2.0*ran1(idum) - 1.0;
      rsq = v1*v1 + v2*v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0*log(rsq) / rsq);
    // Make the Box-Muller transformation to get two normal deviates.
    gset = v1*fac;
    // Set the reinitialization flag.
    iset = 1;
    // Return one deviate and save the other for next time.
    return v2*fac;
  // If we have an extra deviate handy, unset the reinitialization flag and
  // return the available deviate.
  } else {
    iset = 0;
    return gset;
  }
}


/* Wrapper for local implementation of gasdev.

This function stores the result of gasdev in a user-provided variable, in order
to leverage the PETSc error-checking machinery. Users may also directly use
gasdev in functional form.
*/
static PetscErrorCode
Gasdev(long *idum, PetscReal *result)
{
  PetscFunctionBeginUser;
  *result = (PetscReal)gasdev(idum);
  PetscFunctionReturn(PETSC_SUCCESS);
}


/* Adaptation of sobseq from Numerical Recipes, 2nd edition.

This version differs as follows:
- The iv array is uninitialized and there is a new ic array in place of the
  original iv array
- The (*n < 0) block initializes iv from ic.

These changes are based on the sobseq function written by Bernie Vasquez
(adapted from NR 2nd ed.), in the hybrid electromagnetic PIC code developed by
Bernie Vasquez, Harald Kucharek, and Matt Young at UNH circa 2019. The note
there reads "iv is initialized properly on each *n<0 call".
*/
static PetscErrorCode
Sobseq(PetscInt *n, PetscReal x[])
{
  PetscInt j, k, l;
  unsigned long i, im, ipp;
  static unsigned long in;
  static unsigned long ix[MAXDIM+1];
  static unsigned long *iu[MAXBIT+1];
  static unsigned long mdeg[MAXDIM+1]={0, 1, 2, 3, 3, 4, 4};
  static unsigned long ip[MAXDIM+1]={0, 0, 1, 1, 2, 1, 4};
  static unsigned long iv[MAXDIM*MAXBIT+1];
  static unsigned long ic[MAXDIM*MAXBIT+1]={0, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 1, 1, 5, 7, 7, 3, 3, 5, 15, 11, 5, 15, 13, 9};
  static PetscReal fac;

  PetscFunctionBeginUser;

  if (*n < 0) {
    for (j=1; j<=MAXDIM*MAXBIT+1; j++) {
      iv[j] = ic[j];
    }
    for (j=1, k=0; j<=MAXBIT; j++, k += MAXDIM) {
      iu[j] = &iv[k];
    }
    for (k=1; k<=MAXDIM; k++) {
      for (j=1; j<=mdeg[k]; j++) {
        iu[j][k] <<= (MAXBIT-j);
      }
      for (j=mdeg[k]+1; j<=MAXBIT; j++) {
        ipp = ip[k];
        i = iu[j-mdeg[k]][k];
        i ^= (i >> mdeg[k]);
        for (l=mdeg[k]-1; l>=1; l--) {
          if (ipp & 1) i ^= iu[j-1][k];
          ipp >>= 1;
        }
        iu[j][k] = i;
      }
    }
    fac = 1.0 / ((long)1 << MAXBIT);
    in = 0;
  } else {
    im = in++;
    for (j=1; j<=MAXBIT; j++) {
      if (!(im & 1)) break;
      im >>= 1;
    }
    if (j > MAXBIT) {
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "MAXBIT too small in %s", __func__);
    }
    im = (j-1)*MAXDIM;
    for (k=1; k<=PetscMin(*n, MAXDIM); k++) {
      ix[k] ^= iv[im+k];
      x[k] = ix[k]*fac;
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


