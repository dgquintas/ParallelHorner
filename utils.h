#ifndef __UTILS_H
#define __UTILS_H

#include <gmpxx.h>

int GetMemoryFootprint(const mpf_class& r) {
  (void)r;
  return 0;  // whatever
}

int GetMemoryFootprint(const mpz_class& z) {
  return mpz_size(z.get_mpz_t()) * sizeof(mp_limb_t);
}

int GetMemoryFootprint(const mpq_class& q) {
  const mpz_class& num = q.get_num();
  const mpz_class& den = q.get_den();
  return GetMemoryFootprint(num) + GetMemoryFootprint(den);
}

#endif  // __UTILS_H
