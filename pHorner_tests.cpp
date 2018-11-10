#include <gmpxx.h>
#include <memory>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "pHorner.hpp"

template <typename T>
std::vector<T> GetCoefficientsForE(const int n);

template <>
std::vector<mpq_class> GetCoefficientsForE(const int n) {
  std::vector<mpq_class> p(n, mpq_class(1));
  mpz_class running_factorial(1);
  for (int i = 1; i < n; ++i) {
    running_factorial *= i;
    p[i].get_den() = running_factorial;
  }
  return p;
}

class HornerTest : public testing::Test {
 protected:
  virtual void SetUp() {
    p_.push_back(1);
    p_.push_back(2);
    p_.push_back(3);
    p_.push_back(4);

    std::unique_ptr<Partitioner<mpq_class>> partitioner(
        new WidthPartitioner<mpq_class>(0.1));

    horner_.reset(new Horner<mpq_class>(std::move(partitioner)));
  }

  mpq_class powerQ(mpq_class rop, const unsigned long int exp) {
    mpz_pow_ui(rop.get_num_mpz_t(), rop.get_num_mpz_t(), exp);
    mpz_pow_ui(rop.get_den_mpz_t(), rop.get_den_mpz_t(), exp);
    return rop;
  }

  std::vector<mpq_class> p_;
  std::unique_ptr<Horner<mpq_class>> horner_;
};

TEST_F(HornerTest, Iterative) {
  {
    const mpq_class x0 = 1;
    const mpq_class result =
        horner_->Evaluate(p_, x0, Horner<mpq_class>::ITERATIVE);
    const mpq_class expected(10);
    EXPECT_EQ(expected, result);
  }

  {
    const mpq_class x0 = 2.2;
    const mpq_class result =
        horner_->Evaluate(p_, x0, Horner<mpq_class>::ITERATIVE);
    const mpq_class expected(1 + 2 * x0 + 3 * x0 * x0 + 4 * x0 * x0 * x0);
    EXPECT_EQ(expected, result);
  }
}

TEST_F(HornerTest, Parallel) {
  {
    const mpq_class x0 = 1;
    const mpq_class result =
        horner_->Evaluate(p_, x0, Horner<mpq_class>::PARALLEL);
    const mpq_class expected(10);
    EXPECT_EQ(expected, result);
  }

  {
    const mpq_class x0 = 2.2;
    const mpq_class result =
        horner_->Evaluate(p_, x0, Horner<mpq_class>::PARALLEL);
    const mpq_class expected(1 + 2 * x0 + 3 * x0 * x0 + 4 * x0 * x0 * x0);
    EXPECT_EQ(expected, result);
  }
}


TEST_F(HornerTest, E) {
  constexpr auto N = 100;
  const std::vector<mpq_class> p(GetCoefficientsForE<mpq_class>(N));
  const mpq_class x0 = 2.2;
  const mpq_class iter_res =
        horner_->Evaluate(p_, x0, Horner<mpq_class>::ITERATIVE);
  const mpq_class par_res =
        horner_->Evaluate(p_, x0, Horner<mpq_class>::PARALLEL);
  ASSERT_EQ(iter_res, par_res);
}

TEST_F(HornerTest, Sparse) {
  std::vector<int> exponents = {0, 5, 7, 10};
  {
    const mpq_class x0 = 2.2;
    const mpq_class result =
        horner_->hornerIterSparse(p_.begin(), p_.end(), exponents.begin(), x0);
    const mpq_class expected(1 + 2 * powerQ(x0, 5) + 3 * powerQ(x0, 7) +
                             4 * powerQ(x0, 10));
    EXPECT_EQ(expected, result);
  }

  {
    const mpq_class x0 = 1;
    const mpq_class result =
        horner_->hornerIterSparse(p_.begin(), p_.end(), exponents.begin(), x0);
    const mpq_class expected(10);
    EXPECT_EQ(expected, result);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
