#include <array>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include <simd.hpp>

using namespace ASC_HPC;
using std::cout, std::endl;

template <size_t N>
void expect_mask(SIMD<mask64, N> mask, std::array<bool, N> expected, const char* name)
{
  for (size_t i = 0; i < N; ++i) {
    const bool got = bool(mask[i]);
    if (got != expected[i]) {
      std::cerr << name << " mismatch at lane " << i << ": expected "
                << expected[i] << " got " << got << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

void testComparisonsDouble2()
{
  SIMD<double,2> a(1.0, 4.0);
  SIMD<double,2> b(1.0, 2.0);

  expect_mask(a == b, {true, false}, "eq double2");
  expect_mask(a != b, {false, true}, "neq double2");
  expect_mask(a >= b, {true, true}, "ge double2");
  expect_mask(a <= b, {true, false}, "le double2");
  expect_mask(a > b,  {false, true}, "gt double2");
  expect_mask(a < b,  {false, false}, "lt double2");
}

void testComparisonsDouble4()
{
  SIMD<double,4> a(1.0, 2.0, 3.0, 4.0);
  SIMD<double,4> b(0.0, 2.0, 6.0, 1.0);

  expect_mask(a == b, {false, true,  false, false}, "eq double4");
  expect_mask(a != b, {true,  false, true,  true }, "neq double4");
  expect_mask(a >= b, {true,  true,  false, true }, "ge double4");
  expect_mask(a <= b, {false, true,  true,  false}, "le double4");
  expect_mask(a > b,  {true,  false, false, true }, "gt double4");
  expect_mask(a < b,  {false, false, true,  false}, "lt double4");
}

void testTransposeDouble4()
{
  SIMD<double,4> r0(1.0, 2.0, 3.0, 4.0);
  SIMD<double,4> r1(5.0, 6.0, 7.0, 8.0);
  SIMD<double,4> r2(9.0, 10.0, 11.0, 12.0);
  SIMD<double,4> r3(13.0, 14.0, 15.0, 16.0);

  SIMD<double,4> c0, c1, c2, c3;
  transpose(r0, r1, r2, r3, c0, c1, c2, c3);

  auto check = [](SIMD<double,4> v, std::array<double,4> expected, const char* name) {
    std::array<double,4> lanes{};
    v.store(lanes.data());
    for (size_t i = 0; i < 4; ++i) {
      if (lanes[i] != expected[i]) {
        std::cerr << name << " mismatch at lane " << i << ": expected "
                  << expected[i] << " got " << lanes[i] << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  };

  check(c0, {1.0, 5.0, 9.0, 13.0}, "transpose col0");
  check(c1, {2.0, 6.0, 10.0, 14.0}, "transpose col1");
  check(c2, {3.0, 7.0, 11.0, 15.0}, "transpose col2");
  check(c3, {4.0, 8.0, 12.0, 16.0}, "transpose col3");
}

auto func1 (SIMD<double> a, SIMD<double> b)
{
  return a+b;
}

auto func2 (SIMD<double,4> a, SIMD<double,4> b)
{
  return a+3*b;
}

auto func3 (SIMD<double,4> a, SIMD<double,4> b, SIMD<double,4> c)
{
  return fma(a,b,c);
}


auto load (double * p)
{
  return SIMD<double,2>(p);
}

auto loadMask(double *p, SIMD<mask64, 2> m)
{
  return SIMD<double,2>(p, m);
}

auto testSelect (SIMD<mask64,2> m, SIMD<double,2> a, SIMD<double,2> b)
{
  return select (m, a, b);
}

SIMD<double,2> testHSum (SIMD<double,4> a, SIMD<double,4> b)
{
  return hSum(a,b);
}



int main()
{
  testComparisonsDouble2();
  testComparisonsDouble4();
  testTransposeDouble4();
  cout << "SIMD comparison and transpose self-test passed." << endl;

  SIMD<double,4> a(1.,2.,3.,4.);
  SIMD<double,4> b(1.0);
  
  cout << "a = " << a << endl;
  cout << "b = " << b << endl;
  cout << "a+b = " << a+b << endl;

  cout << "HSum(a) = " << hSum(a) << endl;
  cout << "HSum(a,b) = " << hSum(a,b) << endl;

  
  auto sequ = IndexSequence<int64_t, 4>();
  cout << "sequ = " << sequ << endl;
  auto mask = (2 >= sequ);
  cout << "2 >= " << sequ << " = " << mask << endl;

  {
    double a[] = { 10, 10, 10, 10 };
    SIMD<double,4> sa(&a[0], mask);
    cout << "sa = " << sa << endl;
  }

  cout << "select(mask, a, b) = " << select(mask, a,b) << endl;
  
}
