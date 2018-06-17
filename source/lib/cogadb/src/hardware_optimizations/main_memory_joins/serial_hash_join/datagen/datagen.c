/**
 * @file
 *
 * Data generator
 *
 * @author Jens Teubner <jens.teubner@cs.tu-dortmund.de>
 *
 * $Id$
 */

#include "config.h"

#include "datagen/datagen.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Generate a relation where key values are a random permutation
 * the values 1 .. @a num_tuples.
 */
relation_t gen_unique(unsigned long num_tuples) {
  tuple_t *tuples = malloc(num_tuples * sizeof(*tuples));

  if (!tuples) {
    fprintf(stderr, "Error allocating memory (gen_unique ()).\n");
    exit(EXIT_FAILURE);
  }

  /* first populate with a dense sequence 1..num_tuples */
  for (unsigned long i = 0; i < num_tuples; i++) {
    tuples[i].key = i;
    tuples[i].value = random();
  }

  /* then permute */
  for (unsigned long i = num_tuples - 1; i > 0; i--) {
    unsigned long j = random() % num_tuples;
    tuple_t t = tuples[i];
    tuples[i] = tuples[j];
    tuples[j] = t;
  }

  return (struct relation_t){.count = num_tuples, .tuples = tuples};
}

/**
 * Create an alphabet, an array of size @a size with randomly
 * permuted values 0..size-1.
 *
 * @param size alphabet size
 * @return an <code>ht_key_t</code> array with @a size elements;
 *         contains values 0..size-1 in a random permutation; the
 *         return value is malloc'ed, don't forget to free it afterward.
 */
static ht_key_t *gen_alphabet(unsigned long size) {
  ht_key_t *alphabet;

  /* allocate */
  alphabet = malloc(size * sizeof(*alphabet));
  assert(alphabet);

  /* populate */
  for (unsigned long i = 0; i < size; i++) alphabet[i] = i;

  /* permute */
  for (unsigned int i = size - 1; i > 0; i--) {
    unsigned int k = (unsigned long)i * random() / RAND_MAX;
    unsigned int tmp;

    tmp = alphabet[i];
    alphabet[i] = alphabet[k];
    alphabet[k] = tmp;
  }

  return alphabet;
}

/**
 * Generate a lookup table with the cumulated density function
 *
 * (This is derived from code originally written by Rene Mueller.)
 */
static double *gen_zipf_lut(double zipfparam, unsigned long alphabet_size) {
  double scaling_factor;
  double sum;

  double *lut; /**< return value */

  lut = malloc(alphabet_size * sizeof(*lut));
  assert(lut);

  /*
   * Compute scaling factor such that
   *
   *   sum (lut[i], i=1..alphabet_size) = 1.0
   *
   */
  scaling_factor = 0.0;
  for (unsigned long i = 1; i <= alphabet_size; i++)
    scaling_factor += 1.0 / pow(i, zipfparam);

  /*
   * Generate the lookup table
   */
  sum = 0.0;
  for (unsigned long i = 1; i <= alphabet_size; i++) {
    sum += 1.0 / pow(i, zipfparam);
    lut[i - 1] = sum / scaling_factor;
  }

  return lut;
}

relation_t gen_zipf(unsigned long num_tuples, unsigned long alphabet_size,
                    double zipfparam) {
  tuple_t *tuples;

  /* prepare stuff for Zipf generation */
  ht_key_t *alphabet = gen_alphabet(alphabet_size);
  assert(alphabet);

  double *lut = gen_zipf_lut(zipfparam, alphabet_size);
  assert(lut);

  tuples = malloc(num_tuples * sizeof(*tuples));
  assert(tuples);

  for (unsigned long i = 0; i < num_tuples; i++) {
    /* take random number */
    double r = ((double)random()) / RAND_MAX;

    /* binary search in lookup table to determine item */
    unsigned long left = 0;
    unsigned long right = alphabet_size - 1;
    unsigned long m;   /* middle between left and right */
    unsigned long pos; /* position to take */

    if (lut[0] >= r)
      pos = 0;
    else {
      while (right - left > 1) {
        m = (left + right) / 2;

        if (lut[m] < r)
          left = m;
        else
          right = m;
      }

      pos = right;
    }

    tuples[i].key = alphabet[pos];
    tuples[i].value = random();
  }

  free(lut);
  free(alphabet);

  return (struct relation_t){.count = num_tuples, .tuples = tuples};
}
