#include <iostream>
#include <sstream>
#include <chrono>


#include <simd.hpp>

using namespace ASC_HPC;
using namespace std;


/*
  useful functions for matrix-matrix multiplication with vector updates:
  C_i* += Aij Bj* 
 */

// daxpy: y += alpha * x
void daxpy (size_t n, double * px, double * py, double alpha)
{
  SIMD<double> simd_alpha(alpha);
  for (size_t i = 0; i < n; i += SIMD<double>::size())
    {
      SIMD<double> yi(py+i);
      // yi += simd_alpha * SIMD<double> (px+i);
      yi = fma(simd_alpha, SIMD<double> (px+i), yi);
      yi.store(py+i);
    }
}

// multi-daxpy:
// y0 += alpha00 * x0 + alpha01 * x1
// y1 += alpha10 * x0 + alpha11 * x1
void daxpy2x2 (size_t n, double * px0, double * px1,
               double * py0, double * py1, double alpha00, double alpha01, double alpha10, double alpha11)
{
  SIMD<double> simd_alpha00(alpha00);
  SIMD<double> simd_alpha01(alpha01);
  SIMD<double> simd_alpha10(alpha10);
  SIMD<double> simd_alpha11(alpha11);  
  
  for (size_t i = 0; i < n; i += SIMD<double>::size())
    {
      SIMD<double> xi0(px0+i);
      SIMD<double> xi1(px1+i);
      // (SIMD<double>(py0+i)+simd_alpha00*xi0+simd_alpha01*xi1).store(py0+i);
      // (SIMD<double>(py1+i)+simd_alpha10*xi0+simd_alpha11*xi1).store(py1+i);

      SIMD<double> yi0(py0+i);
      yi0 = fma(simd_alpha00, xi0, yi0);
      yi0 = fma(simd_alpha01, xi1, yi0);
      yi0.store(py0+i);
      
      SIMD<double> yi1(py1+i);
      yi1 = fma(simd_alpha10, xi0, yi1);
      yi1 = fma(simd_alpha11, xi1, yi1);
      yi1.store(py1+i);
    }
}


/*
  useful functions for matrix-matrix multiplication with inner products:
  C_iJ = sum* Ai* B*J       where J = ( j,j+1, ... j+SW-1 )
 */

// Inner product
// x ... double vector
// y ... simd vector, with dy distance in doubles
template <size_t SW>
auto InnerProduct (size_t n, double * px, double * py, size_t dy)
{
  SIMD<double,SW> sum{0.0};
  for (size_t i = 0; i < n; i++)
    {
      // sum += px[i] * SIMD<double,SW>(py+i*dy);
      sum = fma(SIMD<double,SW>(px[i]), SIMD<double,SW>(py+i*dy), sum);
    }
  return sum;
}

// generate function to look at assembly code
auto InnerProduct8 (size_t n, double * px, double * py, size_t dy)
{
  return InnerProduct<8> (n, px, py, dy);
}


// Inner product
// x0,x1 ... double vector
// y ... simd vector, with dy distance in doubles
template <size_t SW>
auto InnerProduct2 (size_t n, double * px0, double * px1, double * py, size_t dy)
{
  SIMD<double,SW> sum0{0.0};
  SIMD<double,SW> sum1{0.0};
  for (size_t i = 0; i < n; i++)
    {
      // sum += px[i] * SIMD<double,SW>(py+i*dy);
      SIMD<double,SW> yi(py+i*dy);
      sum0 = fma(SIMD<double,SW>(px0[i]), yi, sum0);
      sum1 = fma(SIMD<double,SW>(px1[i]), yi, sum1);      
    }
  return tuple(sum0, sum1);
}




int main()
{
  int a = 8, b = 2; //Test von Sort2 für ints
  Sort2(a,b); //erwartet a=2, b=8
  std::cout << "Sort2 scalar -> a=" << a << "b=" << b << "\n";
  if (!(a == 2 && b== 8)) {
    std::cerr << "Sort2 scalar FAILED \n";
    return 1;
  }

  SIMD<double,4> va(7, 1, 6, 5);
  SIMD<double,4> vb(2, 8, 4, 2);

  Sort2(va, vb);

  bool ok = (va[0]==2 && va[1]==1 && va[2]==4 && va[3]==2 && vb[0]==7 && vb[1]==8 && vb[2]==6 && vb[3]==5);
  std::cout << "Sort2 SIMD<double,4> -> " << (ok ? "OK" : "FAILED") << "\n";
  if (!ok) return 1;


  int arr[8] = {7,2,5,1,9,0,4,3};     // anderer Name als das frühere 'a'

  BitonicSort<true>(arr, 8);

  bool ok_bitonic = true;              // anderer Name als dein früheres 'ok'
  for (int i = 1; i < 8; ++i) {
    if (arr[i-1] > arr[i]) { ok_bitonic = false; break; }
  }

  std::cout << "BitonicSort scalar -> " << (ok_bitonic ? "OK" : "FAILED") << "\n";
  if (!ok_bitonic) return 1;

  using V = SIMD<double,4>;

  // 8 Vektoren (Länge = 2^k). Jede Lane bekommt eigene Werte damit wir lane-weise sortiert prüfen können.
  V vec[8] = {
    V( 7.0,  1.0,  5.0, -2.0), // V(lane 0, lane 1, lane 2, lane 3)
    V( 2.0,  8.0,  5.0,  9.0),
    V( 5.0,  5.0,  0.0,  0.0),
    V( 1.0,  9.0, -3.0,  4.0),
    V( 9.0,  0.0,  4.0,  3.0),
    V( 0.0, -1.0,  6.0,  1.0),
    V( 4.0,  4.0,  2.0,  8.0),
    V( 3.0,  3.0,  7.0,  2.0),
  };
  
  BitonicSort<true>(vec, 8); // aufwärts sortieren

  bool ok_simd = true; // lane-weise Monotonie prüfen
  for (int lane = 0; lane < 4; ++lane) {
    for (int i = 1; i < 8; ++i) {
      if (vec[i-1][lane] > vec[i][lane]) {
        ok_simd = false;
        break;
      }
    }
    if (!ok_simd) break;
  }

  std::cout << "BitonicSort SIMD<double,4> (asc) -> "
            << (ok_simd ? "OK" : "FAILED") << "\n";
  if (!ok_simd) return 1;

  // lane 0 als sortierte Liste
  std::cout << "sorted lane 0: ";
  for (int i = 0; i < 8; ++i) std::cout << vec[i][0] << " ";
  std::cout << "\n";

  // lane 1
  std::cout << "sorted lane 1: ";
  for (int i = 0; i < 8; ++i) std::cout << vec[i][1] << " ";
  std::cout << "\n";



  cout << "timing daxpy" << endl;  
  for (size_t n = 16; n <= 1024; n*= 2)
    {
      double * px = new double[n];
      double * py = new double[n];
      for (size_t i = 0; i < n; i++)
        {
          px[i] = i;
          py[i] = 2;
        }
  
      auto start = std::chrono::high_resolution_clock::now();

      size_t runs = size_t (1e8 / n) + 1;

      for (size_t i = 0; i < runs; i++)
        daxpy (n, px, py, 2.8);
      
      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration<double>(end-start).count();
        
      cout << "n = " << n << ", time = " << time << " s, GFlops = "
           << (n*runs)/time*1e-9 << endl;
      
      delete [] py;
      delete [] px;
    }


  cout << "timing daxpy 2x2" << endl;
  for (size_t n = 16; n <= 1024; n*= 2)
    {
      double * px0 = new double[n];
      double * py0 = new double[n];
      double * px1 = new double[n];
      double * py1 = new double[n];
      for (size_t i = 0; i < n; i++)
        {
          px0[i] = i;
          py0[i] = 2;
          px1[i] = i;
          py1[i] = 2;
        }
  
      auto start = std::chrono::high_resolution_clock::now();

      size_t runs = size_t (1e8 / (4*n)) + 1;

      for (size_t i = 0; i < runs; i++)
        daxpy2x2 (n, px0, px1, py0, py1, 0.4, 1.3, 2.5, 2.8);
      
      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration<double>(end-start).count();
        
      cout << "n = " << n << "time = " << time << " s, GFlops = "
           << (4*n*runs)/time*1e-9 << endl;
      
      delete [] py0;
      delete [] px0;
      delete [] py1;
      delete [] px1;
    }

  

  constexpr size_t SW=2;
  cout << "timing inner product 1x" << SW << endl;
  for (size_t n = 16; n <= 1024; n*= 2)
    {
      double * px = new double[n];
      double * py = new double[n*SIMD<double,SW>::size()];
      for (size_t i = 0; i < n; i++)
        px[i] = i;
      for (size_t i = 0; i < n*SIMD<double,SW>::size(); i++)
        py[i] = 2;
  
      auto start = std::chrono::high_resolution_clock::now();

      size_t runs = size_t (1e8 / (n*SIMD<double,SW>::size())) + 1;

      SIMD<double,SW> sum{0.0};
      for (size_t i = 0; i < runs; i++)
        sum += InnerProduct<SW> (n, px, py, SIMD<double,SW>::size());
      
      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration<double>(end-start).count();
      cout << "sum = " << sum;
      cout << ", n = " << n << ", time = " << time 
           << " s, GFlops = " << (SIMD<double,SW>::size()*n*runs)/time*1e-9
           << endl;
      
      delete [] py;
      delete [] px;
    }
  
  {
  constexpr size_t SW=4;
  cout << "timing inner product 2x" << SW << endl;
  for (size_t n = 16; n <= 1024; n*= 2)
    {
      double * px0 = new double[n];
      double * px1 = new double[n];
      double * py = new double[n*SW];
      for (size_t i = 0; i < n; i++)
        {
          px0[i] = i;
          px1[i] = 3+i;
        }
      for (size_t i = 0; i < n*SW; i++)
        py[i] = 2;
  
      auto start = std::chrono::high_resolution_clock::now();

      size_t runs = size_t (1e8 / (n*2*SW)) + 1;

      SIMD<double,SW> sum{0.0};
      for (size_t i = 0; i < runs; i++)
        {
          auto [sum0,sum1] = InnerProduct2<SW> (n, px0, px1, py, SW);
          sum += sum0 + sum1;
        }
      
      auto end = std::chrono::high_resolution_clock::now();
      double time = std::chrono::duration<double>(end-start).count();
      cout << "sum = " << sum;
      cout << ", n = " << n << ", time = " << time 
           << " s, GFlops = " << (2*SW*n*runs)/time*1e-9
           << endl;
      
      delete [] py;
      delete [] px0;
      delete [] px1;
    }
  }
  
}
