#ifdef _OPENMP
#include <omp.h>
#else
void omp_set_num_threads(int);
#endif

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>

#include <gmpxx.h>

#include "pHorner.hpp"
#include "partitioners.hpp"

#include "benchmark/benchmark.h"

std::vector<mpq_class> GetCoefficientsQForE(const int n) {
  std::vector<mpq_class> p(n, mpq_class(1));
  mpz_class running_factorial(1);
  for (int i = 1; i < n; ++i) {
    running_factorial *= i;
    p[i].get_den() = running_factorial;
  }
  return p;
}

std::vector<mpq_class> GetCoefficientsQForSin(const int n) {
  std::vector<mpq_class> p(n, mpq_class(0));
  mpz_class running_factorial(1);
  int sign = 1;
  for (int i = 1; i < n; ++i) {
    running_factorial *= i;
    if (i & 0x1) {
      p[i].get_den() = running_factorial;
      p[i].get_num() = mpz_class(sign);
      sign *= -1;
    }
  }
  return p;
}

std::vector<mpf_class> CastQToR(
    std::function<std::vector<mpq_class>(const int n)> generator, const int n,
    int precision) {
  mpf_set_default_prec(precision);
  const std::vector<mpq_class> exp_q(generator(n));
  std::vector<mpf_class> exp_r;
  exp_r.reserve(exp_q.size());
  for (size_t i = 0; i < exp_q.size(); i++) {
    exp_r.emplace_back(mpf_class(exp_q[i]));
  }
  return exp_r;
}

const char* StrApproximation(const mpq_class& q, const int bitprec = 2048) {
  static char str[32];
  mpf_set_default_prec(bitprec);
  const mpf_class res_float(q);
  gmp_snprintf(str, 32, "%.25Fe\n", res_float.get_mpf_t());
  return str;
}

template <int N>
void WidthQ(::benchmark::State& state) {
  mpq_class x0(2.2);
  std::unique_ptr<Partitioner<mpq_class>> partitioner =
      std::unique_ptr<WidthPartitioner<mpq_class>>(
          new WidthPartitioner<mpq_class>(state.range(1)));
  const Horner<mpq_class> horner(std::move(partitioner));
  omp_set_num_threads(state.range(0));
  mpq_class res;
  const std::vector<mpq_class> p(GetCoefficientsQForSin(N));
  while (state.KeepRunning()) {
    res = horner.Evaluate(p, x0);
    ::benchmark::DoNotOptimize(res);
  }
  state.SetLabel(StrApproximation(res));
}

template <int N>
void WidthR(::benchmark::State& state) {
  const int float_precision = 256;
  const int num_threads = state.range(0);
  const int subproblem_percent = state.range(1);
  mpf_class x0r(2.2, float_precision);
  std::unique_ptr<Partitioner<mpf_class>> partitioner =
      std::unique_ptr<Partitioner<mpf_class>>(
          new WidthPartitioner<mpf_class>(subproblem_percent / 100.0));
  const Horner<mpf_class> horner(std::move(partitioner));
  const std::vector<mpf_class> p(
      CastQToR(GetCoefficientsQForE, N, float_precision));

  omp_set_num_threads(num_threads);
  mpf_class res;
  while (state.KeepRunning()) {
    res = horner.Evaluate(p, x0r);
    ::benchmark::DoNotOptimize(res);
  }
  // // Sanity check
  // const mpf_class resIter =
  //     horner.Evaluate(p, x0r, Horner<mpf_class>::ITERATIVE);
  // constexpr double epsilon = 0.0000001;
  // res -= resIter;
  // if (abs(res) > epsilon) {
  //   abort();
  // }
}  //

template <int N>
void IterativeR(::benchmark::State& state) {
  const size_t precision = 256;
  mpf_class x0r(2.2, precision);
  const Horner<mpf_class> horner(nullptr);
  const std::vector<mpf_class> p(CastQToR(GetCoefficientsQForE, N, precision));
  mpf_class res;
  while (state.KeepRunning()) {
    res = horner.Evaluate(p, x0r, Horner<mpf_class>::ITERATIVE);
    ::benchmark::DoNotOptimize(res);
  }
}

template <int MAX_THREADS, int MIN_PERCENT, int PERCENT_INCR>
void BuildThreadsRange(benchmark::internal::Benchmark* b) {
  for (int t = 1; t <= MAX_THREADS; ++t) {
    const int MAX_PERCENT = 100.0;
    for (float pct = MIN_PERCENT; pct <= MAX_PERCENT; pct += PERCENT_INCR) {
      b->ArgPair(t, pct);
    }
  }
};

template <int MIN_FLOAT_PRECISION, int MAX_FLOAT_PRECISION,
          int FLOAT_PRECISION_INCR, int MIN_SIZE, int MAX_SIZE, int SIZE_INCR>
void BuildPrecisionRange(benchmark::internal::Benchmark* b) {
  for (int i = MIN_FLOAT_PRECISION; i <= MAX_FLOAT_PRECISION;
       i += FLOAT_PRECISION_INCR)
    for (int j = MIN_SIZE; j <= MAX_SIZE; j += SIZE_INCR) b->ArgPair(i, j);
};

template <int N_INI, int N_MAX, int N_INCR>
void BuildNIterativeRange(benchmark::internal::Benchmark* b) {
  for (int i = N_INI; i <= N_MAX; i += N_INCR) {
    b->Arg(i);
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ::benchmark::Initialize(&argc, argv);

  auto ThreadsRange = BuildThreadsRange<1, 100, 1>;

  BENCHMARK_TEMPLATE(WidthR, 100)->Apply(ThreadsRange)->UseRealTime();
  BENCHMARK_TEMPLATE(WidthR, 200)->Apply(ThreadsRange)->UseRealTime();
  BENCHMARK_TEMPLATE(WidthR, 500)->Apply(ThreadsRange)->UseRealTime();
  BENCHMARK_TEMPLATE(WidthR, 1000)->Apply(ThreadsRange)->UseRealTime();
  BENCHMARK_TEMPLATE(WidthR, 2000)->Apply(ThreadsRange)->UseRealTime();
  BENCHMARK_TEMPLATE(WidthR, 3000)->Apply(ThreadsRange)->UseRealTime();
  BENCHMARK_TEMPLATE(WidthR, 4000)->Apply(ThreadsRange)->UseRealTime();
  BENCHMARK_TEMPLATE(WidthR, 5000)->Apply(ThreadsRange)->UseRealTime();

  BENCHMARK_TEMPLATE(IterativeR, 100)->UseRealTime();
  BENCHMARK_TEMPLATE(IterativeR, 200)->UseRealTime();
  BENCHMARK_TEMPLATE(IterativeR, 500)->UseRealTime();
  BENCHMARK_TEMPLATE(IterativeR, 1000)->UseRealTime();
  BENCHMARK_TEMPLATE(IterativeR, 2000)->UseRealTime();
  BENCHMARK_TEMPLATE(IterativeR, 3000)->UseRealTime();
  BENCHMARK_TEMPLATE(IterativeR, 4000)->UseRealTime();
  BENCHMARK_TEMPLATE(IterativeR, 5000)->UseRealTime();
  ::benchmark::RunSpecifiedBenchmarks();
}
