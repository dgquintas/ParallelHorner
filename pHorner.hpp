#ifndef __HORNER_H
#define __HORNER_H


#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <vector>

#include <gmpxx.h>
#include "gtest/gtest.h"

#ifdef _OPENMP
#include <omp.h>
#else
// just declare the omp_ functions to avoid annoying errors.
double omp_get_wtime();
int omp_get_thread_num();
int omp_get_max_threads();
#endif

#include "partitioners.hpp"
#include "utils.h"

template <typename T>
class Horner {
 public:
  Horner(std::unique_ptr<Partitioner<T>> partitioner);
  enum Method { ITERATIVE, PARALLEL };
  T Evaluate(const std::vector<T>& p, const T& x0,
             const Method m = PARALLEL) const;

 private:
  T hornerIter(CoefficientIterator<T> start, CoefficientIterator<T> end,
               const T& x0) const;

  using ExponentsIterator = std::vector<int>::const_iterator;
  T hornerIterSparse(CoefficientIterator<T> start, CoefficientIterator<T> end,
                     ExponentsIterator start_e, const T& x0) const;

  T pHorner(const std::vector<T>& p, const T& x0) const;

  const std::unique_ptr<Partitioner<T>> partitioner_;

  FRIEND_TEST(HornerTest, Sparse);
};

//
// Implementation
//
namespace {
void PrintQ(const mpq_class& q) {
  mpf_set_default_prec(1024);
  const mpf_class res_float(q);
  gmp_printf("%.Ff\n", res_float.get_mpf_t());
}

void Power(mpq_class* const rop, const unsigned long int exp) {
  mpz_pow_ui(rop->get_num_mpz_t(), rop->get_num_mpz_t(), exp);
  mpz_pow_ui(rop->get_den_mpz_t(), rop->get_den_mpz_t(), exp);
}

void Power(mpf_class* const rop, const unsigned long int exp) {
  mpf_pow_ui(rop->get_mpf_t(), rop->get_mpf_t(), exp);
}
}  // namespace

template <typename T>
Horner<T>::Horner(std::unique_ptr<Partitioner<T>> partitioner)
    : partitioner_(std::move(partitioner)) {}

template <typename T>
T Horner<T>::Evaluate(const std::vector<T>& p, const T& x0,
                      const Method m) const {
  return m == ITERATIVE ? hornerIter(p.begin(), p.end(), x0) : pHorner(p, x0);
}

/** Iterative version.
 *
 * Evaluates @arg p for @arg x0 for the coefficients between
 * @arg a (lowest coefficient) and @arg b.
 *
 * @param p the polynomial.
 * @param x0 the value we are evaluating it for.
 * @param a index (wrt @arg p) of the constant
 * @param b index (wrt @arg p) of the highest order coefficient
 * */
template <typename T>
T Horner<T>::hornerIter(CoefficientIterator<T> start,
                        CoefficientIterator<T> end, const T& x0) const {
#ifndef NDEBUG
  int mem_accessed = 0;
#endif
  CHECK(start <= end);
  T res(0);
  for (auto it = end - 1; it > start; --it) {
    res = (res + *it) * x0;
#ifndef NDEBUG
    mem_accessed += GetMemoryFootprint(*it);
#endif
  }
  res += *start;
#ifndef NDEBUG
  mem_accessed += GetMemoryFootprint(res);
  VLOG(2) << "ITER size: " << end - start
          << "; mem = " << mem_accessed / 1024.0;
#endif
  return res;
}

/**
 *
 * Pass in a sequence of pairs (a_k, k) for all a_k non-zero coefficients
 * associated with x^k.
 *
 */
template <typename T>
T Horner<T>::hornerIterSparse(CoefficientIterator<T> start,
                              CoefficientIterator<T> end,
                              ExponentsIterator start_e, const T& x0) const {
  if (start == end) {
    return T(0);
  }

  const auto ranges = partitioner_->Partition(start, end);
  const int n = ranges.size();
  if (VLOG_IS_ON(2)) {
    for (int i = 0; i < n; ++i) {
      VLOG(2) << "Part [" << i + 1 << "/" << n << "] (" << ranges[i].first
              << ", " << ranges[i].second << ")";
    }
  }

  auto base_fn = [](CoefficientIterator<T> start, CoefficientIterator<T> end,
                    ExponentsIterator start_e, const T& x0) {
    // Calculate the differences between the powers.
    std::vector<int> power_diffs;
    power_diffs.reserve(end - start);
    power_diffs.push_back(*(start_e++));
    for (auto it = start + 1; it < end; ++start_e, ++it) {
      power_diffs.push_back(*start_e - *(start_e - 1));
    }
    power_diffs.push_back(-1);  // sentinel

    struct DiffCount {
      int diff;
      size_t times;
    };
    std::vector<DiffCount> diff_counts;
    diff_counts.reserve(power_diffs.size());
    size_t i = 0;
    size_t j = 1;
    while (j < power_diffs.size()) {
      while (power_diffs[i] == power_diffs[j++])
        ;
      diff_counts.push_back(DiffCount{power_diffs[i], j - i - 1});
      i = j - 1;
    }

    // Calculate x0^{power_diffs}
    std::vector<T> powers(diff_counts.size(), x0);
    for (size_t i = 0; i < diff_counts.size(); ++i) {
      Power(&(powers[i]), diff_counts[i].diff);
    }

    // This is simply the iterative Horner method using already calculated
    // powers of x0.
    T res(0);
    size_t current_power_times = 0;
    int current_power_idx = powers.size();
    for (auto it = end - 1; it >= start; --it) {
      assert(!diff_counts.empty());
      assert(!powers.empty());
      if (current_power_times == 0) {
        --current_power_idx;
        current_power_times = diff_counts[current_power_idx].times;
        assert(current_power_times > 0);
      }
      res = (res + *it) * powers[current_power_idx];
      --current_power_times;
    }
    assert(current_power_idx == 0);
    assert(current_power_times == 0);
    return res;
  };

  std::vector<T> results(ranges.size());
#pragma omp parallel for default(shared) schedule(dynamic)
  for (int i = 0; i < n; ++i) {
    const CoefficientIterator<T> range_start = start + ranges[i].first;
    const CoefficientIterator<T> range_end = start + ranges[i].second + 1;

    const ExponentsIterator range_start_e = start_e + ranges[i].first;

    // No need to be recursive: the partitioning is in principle generating
    // problem sizes optimal for the base case.
    results[i] = base_fn(range_start, range_end, range_start_e, x0);
  }

  return std::accumulate(results.begin(), results.end(), T(0), std::plus<T>());
}

/** Parallel version.
 *
 * Each individual section uses hornerIter.
 *
 * @param p: polynomial's coefficients
 * @param n: polynomial's degree
 * @param x0: x's value
 * @param m: number of sections FIXME
 */
template <typename T>
T Horner<T>::pHorner(const std::vector<T>& p, const T& x0) const {
  const std::vector<IndexRange> ranges(
      partitioner_->Partition(p.begin(), p.end()));

  std::vector<T> subproblems(ranges.size());
  std::vector<int> exponents(ranges.size());
#pragma omp parallel for default(shared) schedule(dynamic)
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto start = p.begin() + ranges[i].first;
    const auto end = p.begin() + ranges[i].second + 1;
    subproblems[i] = hornerIter(start, end, x0);
    exponents[i] = ranges[i].first;
  }
  return hornerIterSparse(subproblems.begin(), subproblems.end(),
                          exponents.begin(), x0);
}

#endif  // __HORNER_H
