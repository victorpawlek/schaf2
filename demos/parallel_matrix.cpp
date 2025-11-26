#include <iostream>
#include <sstream>
#include <atomic>
#include <cmath>
#include <chrono>
#include <thread>

#include <simd.hpp>
#include <simd_matrix.hpp>
#include <taskmanager.hpp>
#include <timer.hpp>

using namespace ASC_HPC;
using std::cout, std::endl;

namespace ASC_HPC {

    // Parallel matmul: A(M×K row-major) * B(K×N column-major) = C(M×N row-major)
    template <typename T, size_t M, size_t K, size_t N>
    void matmul_parallel_rowwise(
        const SimdMatrix<T,M,K,StorageOrder::RowMajor> &A,
        const SimdMatrix<T,K,N,StorageOrder::ColumnMajor> &B,
        SimdMatrix<T,M,N,StorageOrder::RowMajor> &C)
    {
      // one task per row of C
      auto func = [&](int nr, int size)
      {

        const int row = nr;                 // 0 .. M-1
        if (row >= (int)M) return;          // safety

        // <<< NEW: trace this row task on whatever thread runs it >>>
        static Timer t_row("parallel matmul", {1, 0, 0});  // green, for example
        RegionTimer reg_row(t_row);
  
        auto Ai = A.row_vector(row);        // SIMD<T,K>
  
        for (size_t j = 0; j < N; ++j)
        {
          auto Bj = B.col_vector(j);        // SIMD<T,K>
          auto prod = Ai * Bj;              // elementwise multiply
          T sum = hSum(prod);               // horizontal sum
          C.set((size_t)row, j, sum);
        }
      };
  
      RunParallel((int)M, func);
    }
  
  } 
  
  template <size_t N>
  void benchmark_size()
  {
      using T    = double;
      using MatA = SimdMatrix<T,N,N,StorageOrder::RowMajor>;
      using MatB = SimdMatrix<T,N,N,StorageOrder::ColumnMajor>;
      using MatC = SimdMatrix<T,N,N,StorageOrder::RowMajor>;
  
      // allocate on heap to avoid stack overflow for large N
      auto A = std::make_unique<MatA>();
      auto B = std::make_unique<MatB>();
      auto C = std::make_unique<MatC>(MatC::Zero());
  
      // init matrices with some deterministic values
      for (size_t i = 0; i < N; ++i)
          for (size_t j = 0; j < N; ++j)
          {
              A->set(i, j, T(i + j + 1));
              B->set(i, j, T(i - j + 2));
          }
  
      // number of runs chosen so total flops ~ 1e9-ish
      std::size_t runs_serial   = std::max<std::size_t>(
          std::size_t(1),
          std::size_t(1e9 / (2.0 * N * N * N))
      );
      std::size_t runs_parallel = runs_serial;
  
      volatile T sink = 0.0;
  
      std::cout << "\n=== N = " << N << " ===\n";
  
      // ----------------- serial matmul -------------------------
      {
        static Timer t_serial("serial matmul", {0, 0, 1});  // blue
        RegionTimer reg_serial(t_serial);

          auto start = std::chrono::high_resolution_clock::now();
  
          for (std::size_t r = 0; r < runs_serial; ++r)
          {
              auto Ctmp = (*A) * (*B);  // your existing matmul
              sink += Ctmp(0,0);
          }
  
          auto end  = std::chrono::high_resolution_clock::now();
          double time = std::chrono::duration<double>(end - start).count();
  
          double flops   = 2.0 * double(N) * double(N) * double(N) * double(runs_serial);
          double gflops  = flops / time * 1e-9;
  
          std::cout << "serial:    runs = " << runs_serial
                    << ", time = " << time << " s"
                    << ", GFlops = " << gflops << std::endl;
      }
  
      // ----------------- parallel matmul -----------------------
      {
        // static Timer t_parallel("parallel matmul", {1, 0, 0});  // red
        // RegionTimer reg_parallel(t_parallel);

          auto &Cref = *C; // just an alias
  
          auto start = std::chrono::high_resolution_clock::now();
  
          for (std::size_t r = 0; r < x ; ++r)
          {
              matmul_parallel_rowwise<T,N,N,N>(*A, *B, Cref);
              sink += Cref(0,0);
          }
  
          auto end  = std::chrono::high_resolution_clock::now();
          double time = std::chrono::duration<double>(end - start).count();
  
          double flops   = 2.0 * double(N) * double(N) * double(N) * double(runs_parallel);
          double gflops  = flops / time * 1e-9;
  
          std::cout << "parallel:  runs = " << runs_parallel
                    << ", time = " << time << " s"
                    << ", GFlops = " << gflops << std::endl;
      }
  
      if (sink == T(123456789)) // keep sink “used”
          std::cout << "sink = " << sink << std::endl;
  }
  
  
  // ------------------------------------------------------------
  
  int main()
  {
    timeline = std::make_unique<TimeLine>("parallel_matrix.trace");
    
      // start worker threads once
      int num_threads = (int)std::thread::hardware_concurrency();
      if (num_threads <= 0) num_threads = 4;
      cout << "Using " << num_threads << " worker threads\n";
      StartWorkers(num_threads);
  
      cout << "Benchmarking serial vs parallel matmul (double, N x N)\n";
  
      benchmark_size<32>();
      benchmark_size<64>();
      benchmark_size<128>();
      benchmark_size<256>();
  
      StopWorkers();
      return 0;
  }