#include <iostream>
#include <sstream>
#include <atomic>
#include <cmath>

#include <simd.hpp>
#include <simd_matrix.hpp>

using namespace ASC_HPC;

// ---- helpers ---------------------------------------------------

bool approxEqual(double a, double b, double eps = 1e-9)
{
  return std::abs(a - b) <= eps;
}

template<typename Mat>
bool matricesApproxEqual(const Mat& A, const Mat& B, double eps = 1e-9)
{
  for (size_t r = 0; r < Mat::rows; ++r)
    for (size_t c = 0; c < Mat::cols; ++c)
      if (!approxEqual(A(r,c), B(r,c), eps))
        return false;
  return true;
}

// scalar reference matmul (row-major logical access via operator())
template<typename T, size_t M, size_t K, size_t N, StorageOrder OA, StorageOrder OB, StorageOrder OC>
SimdMatrix<T,M,N,OC> scalarMatmul(const SimdMatrix<T,M,K,OA>& A,
                                  const SimdMatrix<T,K,N,OB>& B)
{
  SimdMatrix<T,M,N,OC> C = SimdMatrix<T,M,N,OC>::Zero();
  for (size_t i = 0; i < M; ++i)
    for (size_t j = 0; j < N; ++j)
    {
      T sum = T(0);
      for (size_t k = 0; k < K; ++k)
        sum += A(i,k) * B(k,j);
      C.set(i,j,sum);
    }
  return C;
}

// ---- tests -----------------------------------------------------

bool testRowMajorAccess()
{
  using Mat = SimdMatrix<double, 2, 3, StorageOrder::RowMajor>;
  Mat A = { 1, 2, 3,
            4, 5, 6 };

  bool ok = true;
  ok &= (A(0,0) == 1);
  ok &= (A(0,1) == 2);
  ok &= (A(0,2) == 3);
  ok &= (A(1,0) == 4);
  ok &= (A(1,1) == 5);
  ok &= (A(1,2) == 6);

  // test set()
  A.set(0,1, 42.0);
  ok &= (A(0,1) == 42.0);

  return ok;
}

bool testColumnMajorAccess()
{
  using Mat = SimdMatrix<double, 2, 3, StorageOrder::ColumnMajor>;
  // initializer_list is still in *logical row-major* order:
  Mat A = { 1, 2, 3,
            4, 5, 6 };

  bool ok = true;
  ok &= (A(0,0) == 1);
  ok &= (A(0,1) == 2);
  ok &= (A(0,2) == 3);
  ok &= (A(1,0) == 4);
  ok &= (A(1,1) == 5);
  ok &= (A(1,2) == 6);

  // test set()
  A.set(1,2, -1.0);
  ok &= (A(1,2) == -1.0);

  return ok;
}

bool testIdentity()
{
  using Mat4RM = SimdMatrix<double,4,4,StorageOrder::RowMajor>;
  auto I = Mat4RM::Identity();

  bool ok = true;
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = 0; j < 4; ++j)
    {
      double expected = (i == j) ? 1.0 : 0.0;
      ok &= approxEqual(I(i,j), expected);
    }
  return ok;
}

bool testMatMulVsScalar()
{
  using A_t = SimdMatrix<double,3,5,StorageOrder::RowMajor>;
  using B_t = SimdMatrix<double,5,4,StorageOrder::ColumnMajor>;
  using C_t = SimdMatrix<double,3,4,StorageOrder::RowMajor>;

  // some arbitrary values
  A_t A = { 1, 2, 3, 4,  5,
            6, 7, 8, 9, 10,
            11,12,13,14,15 };

  B_t B = { 1,  0,  2,  0,
            0,  3,  0,  4,
            5,  0,  6,  0,
            0,  7,  0,  8,
            9, 10, 11, 12 };

  // SIMD matmul (your operator*)
  C_t C_simd = A * B;

  // scalar reference
  C_t C_ref  = scalarMatmul<double,3,5,4,StorageOrder::RowMajor,StorageOrder::ColumnMajor,StorageOrder::RowMajor>(A,B);

  return matricesApproxEqual(C_simd, C_ref, 1e-9);
}

bool testIdentityMultiplication()
{
  using Mat4RM = SimdMatrix<double,4,4,StorageOrder::RowMajor>;
  using Mat4CM = SimdMatrix<double,4,4,StorageOrder::ColumnMajor>;

  Mat4RM A = { 1,  2,  3,  4,
               5,  6,  7,  8,
               9, 10, 11, 12,
               13,14, 15, 16 };

  auto I_RM = Mat4RM::Identity();
  auto I_CM = Mat4CM::Identity();

  // A * I_CM (RowMajor * ColMajor) should equal A
  Mat4RM right = A * I_CM;

  // I_RM * A (RowMajor * ColMajor??) — careful: our operator* expects:
  // SimdMatrix<T,M,K,OA> * SimdMatrix<T,K,N,OB>
  // To get SIMD fast-path, use row-major * col-major.
  // For I * A, let’s just multiply using scalar reference to avoid layout mismatch mess

  Mat4RM left = scalarMatmul<double,4,4,4,StorageOrder::RowMajor,StorageOrder::RowMajor,StorageOrder::RowMajor>(I_RM, A);

  bool ok = true;
  ok &= matricesApproxEqual(right, A);
  ok &= matricesApproxEqual(left,  A);

  return ok;
}

// ---- main -----------------------------------------------------

int main()
{
  bool ok = true;

  ok &= testRowMajorAccess();
  std::cout << "RowMajor access test: " << (ok ? "OK" : "FAIL") << "\n";

  bool ok2 = testColumnMajorAccess();
  std::cout << "ColumnMajor access test: " << (ok2 ? "OK" : "FAIL") << "\n";
  ok &= ok2;

  bool ok3 = testIdentity();
  std::cout << "Identity test: " << (ok3 ? "OK" : "FAIL") << "\n";
  ok &= ok3;

  bool ok4 = testMatMulVsScalar();
  std::cout << "MatMul vs scalar test: " << (ok4 ? "OK" : "FAIL") << "\n";
  ok &= ok4;

  bool ok5 = testIdentityMultiplication();
  std::cout << "Identity * A / A * I test: " << (ok5 ? "OK" : "FAIL") << "\n";
  ok &= ok5;

  std::cout << "\nOVERALL: " << (ok ? "ALL TESTS PASSED 🎉" : "SOME TESTS FAILED 💥") << "\n";

  return ok ? 0 : 1;
}