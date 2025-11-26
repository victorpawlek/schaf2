#ifndef SIMD_MATRIX
#define SIMD_MATRIX

#include<iostream>
#include<string>
#include<memory>
#include <array>
#include <cstdint>
#include <cmath>

#include "simd.hpp"


namespace ASC_HPC {

enum class StorageOrder {
    RowMajor,
    ColumnMajor
};
  
template <typename T, size_t Rows, size_t Cols, StorageOrder Order = StorageOrder::RowMajor>
class SimdMatrix {
    public:
        using value_type = T;
        static constexpr size_t rows = Rows;
        static constexpr size_t cols = Cols;
        static constexpr StorageOrder order = Order;
        static constexpr bool row_major = (Order == StorageOrder::RowMajor);

        // Each *contiguous* row (if row-major) or column (if column-major)
        // is stored as one SIMD vector:
        using major_simd_type = SIMD<T, row_major ? Cols : Rows>;

    private:
        // If row-major: m_data[r] is row r (SIMD<T,Cols>)
        // If col-major: m_data[c] is column c (SIMD<T,Rows>)
        std::array<major_simd_type, row_major ? Rows : Cols> m_data;

    public:
    // --- constructors ------------------------------------------------

        // default: zero
        SimdMatrix()
        {
        for (auto &v : m_data)
            v = major_simd_type(T(0));
        }
    
        // fill with scalar
        explicit SimdMatrix(T v)
        {
        for (auto &x : m_data)
            x = major_simd_type(v);
        }
    
        // construct from flat initializer list (row-major order of values)
        SimdMatrix(std::initializer_list<T> vals)
        {
        // First, fill as row-major in a scalar buffer
        std::array<T, Rows * Cols> buf{};
        {
            auto it = vals.begin();
            for (size_t i = 0; i < Rows * Cols; ++i) {
            if (it != vals.end()) buf[i] = *it++;
            else                  buf[i] = T(0);
            }
        }
    
        // Then store into SIMD layout depending on StorageOrder
        if constexpr (row_major) {
            for (size_t r = 0; r < Rows; ++r) {
            std::array<T, Cols> row_vals{};
            for (size_t c = 0; c < Cols; ++c)
                row_vals[c] = buf[r * Cols + c];
            m_data[r] = major_simd_type(row_vals);
            }
        } else {
            for (size_t c = 0; c < Cols; ++c) {
            std::array<T, Rows> col_vals{};
            for (size_t r = 0; r < Rows; ++r)
                col_vals[r] = buf[r * Cols + c];
            m_data[c] = major_simd_type(col_vals);
            }
        }
        }
  
    // --- basic named constructors ------------------------------------

        static SimdMatrix Zero()
        {
          return SimdMatrix(T(0));
        }
      
        static SimdMatrix Identity()
        {
          static_assert(Rows == Cols, "Identity only defined for square matrices");
          SimdMatrix M = Zero();
          for (size_t i = 0; i < Rows; ++i)
            M.set(i, i, T(1));
          return M;
        }

    // --- element access ----------------------------------------------

        // scalar read
        T operator()(size_t r, size_t c) const
        {
          if constexpr (row_major) {
            return m_data[r][c];
          } else {
            return m_data[c][r];
          }
        }
      
    // scalar write (not super-fast, but convenient)
        void set(size_t r, size_t c, T v)
        {
          if constexpr (row_major) {
            std::array<T, Cols> row_vals{};
            for (size_t j = 0; j < Cols; ++j)
              row_vals[j] = m_data[r][j];
            row_vals[c] = v;
            m_data[r] = major_simd_type(row_vals);
          } else {
            std::array<T, Rows> col_vals{};
            for (size_t i = 0; i < Rows; ++i)
              col_vals[i] = m_data[c][i];
            col_vals[r] = v;
            m_data[c] = major_simd_type(col_vals);
          }
        }
      
        // SIMD view of a full row as SIMD<T,Cols>
        SIMD<T, Cols> row_vector(size_t r) const
        {
          if constexpr (row_major) {
            return m_data[r];          // already SIMD<T,Cols>
          } else {
            std::array<T, Cols> vals{};
            for (size_t c = 0; c < Cols; ++c)
              vals[c] = (*this)(r, c); // gather
            return SIMD<T, Cols>(vals);
          }
        }
      
        // SIMD view of a full column as SIMD<T,Rows>
        SIMD<T, Rows> col_vector(size_t c) const
        {
          if constexpr (!row_major) {
            return m_data[c];          // already SIMD<T,Rows>
          } else {
            std::array<T, Rows> vals{};
            for (size_t r = 0; r < Rows; ++r)
              vals[r] = (*this)(r, c); // gather
            return SIMD<T, Rows>(vals);
          }
        }
      
        // access to underlying SIMD "major" line
        major_simd_type & major(size_t i)             { return m_data[i]; }
        const major_simd_type & major(size_t i) const { return m_data[i]; }
      };
      
      
      // ====================== free functions ================================
      
      // ostream helper
      template <typename T, size_t R, size_t C, StorageOrder O>
      std::ostream & operator<<(std::ostream &os,
                                const SimdMatrix<T,R,C,O> &M)
      {
        for (size_t r = 0; r < R; ++r) {
          os << "[ ";
          for (size_t c = 0; c < C; ++c) {
            os << M(r,c);
            if (c + 1 < C) os << ", ";
          }
          os << " ]\n";
        }
        return os;
      }
      
      
      // matrix + matrix
      template <typename T, size_t R, size_t C, StorageOrder O>
      SimdMatrix<T,R,C,O>
      operator+(const SimdMatrix<T,R,C,O> &A,
                const SimdMatrix<T,R,C,O> &B)
      {
        SimdMatrix<T,R,C,O> Cmat;
        for (size_t i = 0; i < (O == StorageOrder::RowMajor ? R : C); ++i)
          Cmat.major(i) = A.major(i) + B.major(i);
        return Cmat;
      }
      
      // matrix - matrix
      template <typename T, size_t R, size_t C, StorageOrder O>
      SimdMatrix<T,R,C,O>
      operator-(const SimdMatrix<T,R,C,O> &A,
                const SimdMatrix<T,R,C,O> &B)
      {
        SimdMatrix<T,R,C,O> Cmat;
        for (size_t i = 0; i < (O == StorageOrder::RowMajor ? R : C); ++i)
          Cmat.major(i) = A.major(i) - B.major(i);
        return Cmat;
      }
      
      // scalar * matrix
      template <typename T, size_t R, size_t C, StorageOrder O>
      SimdMatrix<T,R,C,O>
      operator*(T s, const SimdMatrix<T,R,C,O> &A)
      {
        SimdMatrix<T,R,C,O> M;
        for (size_t i = 0; i < (O == StorageOrder::RowMajor ? R : C); ++i)
          M.major(i) = A.major(i) * s;
        return M;
      }
      
      // matrix * scalar
      template <typename T, size_t R, size_t C, StorageOrder O>
      inline SimdMatrix<T,R,C,O>
      operator*(const SimdMatrix<T,R,C,O> &A, T s)
      {
        return s * A;
      }
      
      // matrix * SIMD-vector (column vector)
      template <typename T, size_t R, size_t C, StorageOrder O>
      SIMD<T,R>
      operator*(const SimdMatrix<T,R,C,O> &A, const SIMD<T,C> &x)
      {
        SIMD<T,R> y(T(0));
      
        if constexpr (O == StorageOrder::RowMajor) {
          // y_i = dot(row_i, x)
          for (size_t r = 0; r < R; ++r) {
            auto prod = A.row_vector(r) * x;
            T sum = hSum(prod);
            y = select(SIMD<mask64,R>(IndexSequence<int64_t,R>(int64_t(r)) == IndexSequence<int64_t,R>()),
                       SIMD<T,R>(sum), y); // (hacky; for simple use you might just build an array)
          }
        } else {
          // y = A * x = sum_j x_j * col_j
          // start with zero vector
          y = SIMD<T,R>(T(0));
          for (size_t c = 0; c < C; ++c) {
            auto col = A.col_vector(c);
            // need a scalar x_c
            T xc = x[c];
            y += xc * col;
          }
        }
      
        return y;
      }
      
      // matrix * matrix (general, but best if A row-major and B col-major)
      template <typename T,
                size_t M, size_t K, size_t N,
                StorageOrder OA, StorageOrder OB, StorageOrder OC = StorageOrder::RowMajor>
      SimdMatrix<T,M,N,OC>
      matmul(const SimdMatrix<T,M,K,OA> &A,
             const SimdMatrix<T,K,N,OB> &B)
      {
        SimdMatrix<T,M,N,OC> C = SimdMatrix<T,M,N,OC>::Zero();
      
        if constexpr (OA == StorageOrder::RowMajor && OB == StorageOrder::ColumnMajor) {
          // Fast path: A rows and B columns are contiguous SIMD vectors of size K
          for (size_t i = 0; i < M; ++i) {
            auto Ai = A.row_vector(i);      // SIMD<T,K>
            for (size_t j = 0; j < N; ++j) {
              auto Bj = B.col_vector(j);    // SIMD<T,K>
              auto prod = Ai * Bj;
              T sum = hSum(prod);
              C.set(i, j, sum);
            }
          }
        } else {
          // Generic fallback (gathers): scalar triple loop
          for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j) {
              T sum = T(0);
              for (size_t k = 0; k < K; ++k)
                sum += A(i,k) * B(k,j);
              C.set(i,j,sum);
            }
        }
      
        return C;
      }
      
      // convenient operator* using matmul, output row-major by default
      template <typename T,
                size_t M, size_t K, size_t N,
                StorageOrder OA, StorageOrder OB>
      auto operator*(const SimdMatrix<T,M,K,OA> &A,
                     const SimdMatrix<T,K,N,OB> &B)
      {
        return matmul<T,M,K,N,OA,OB>(A,B);
      }


    template <typename T, size_t R, size_t C, StorageOrder Major>
    auto transpose(const SimdMatrix<T,R,C,Major> &A)
    {
        // result is C x R, with opposite major order
        constexpr StorageOrder NewMajor =
            (Major == StorageOrder::RowMajor) ?
            StorageOrder::ColumnMajor :
            StorageOrder::RowMajor;

        SimdMatrix<T, C, R, NewMajor> B = SimdMatrix<T, C, R, NewMajor>::Zero();

        for (size_t r = 0; r < R; ++r)
            for (size_t c = 0; c < C; ++c)
                B.set(c, r, A(r, c));

        return B;
    }
};


#ifdef __AVX__
#include "simd_avx.hpp"
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include "simd_arm64.hpp"
#endif



#endif