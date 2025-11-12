#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "simd.hpp"

using clk = std::chrono::high_resolution_clock;
using sec = std::chrono::duration<double>;

// handy alias
template<size_t W>
using vdouble = ASC_HPC::SIMD<double, W>;

template<size_t W>
static void simd_sincos_array(double* in, double* out_s, double* out_c, size_t n)
{
  using namespace ASC_HPC;
  size_t i = 0;
  for (; i + W <= n; i += W) {
    vdouble<W> x(in + i);
    auto [s, c] = sincos(x);
    s.store(out_s + i);
    c.store(out_c + i);
  }
  // tail (scalar)
  for (; i < n; ++i) {
    auto sc = std::sin(in[i]);
    out_s[i] = sc;
    out_c[i] = std::cos(in[i]);   // use std::cos to avoid extra sincos calls here
  }
}

static void scalar_sincos_array(const double* in, double* out_s, double* out_c, size_t n)
{
  for (size_t i = 0; i < n; ++i) {
    out_s[i] = std::sin(in[i]);
    out_c[i] = std::cos(in[i]);
  }
}

template<size_t W>
static void accuracy_check(size_t n, double range_min, double range_max)
{
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> dist(range_min, range_max);

  std::vector<double> x(n), s_std(n), c_std(n), s_simd(n), c_simd(n);
  for (auto& xi : x) xi = dist(rng);

  scalar_sincos_array(x.data(), s_std.data(), c_std.data(), n);
  simd_sincos_array<W>(x.data(), s_simd.data(), c_simd.data(), n);

  double max_abs_err_s = 0.0, max_abs_err_c = 0.0;
  double max_rel_err_s = 0.0, max_rel_err_c = 0.0;

  auto rel = [](double a, double b) {
    double denom = std::max(1.0, std::abs(b));
    return std::abs(a - b) / denom;
  };

  for (size_t i = 0; i < n; ++i) {
    max_abs_err_s = std::max(max_abs_err_s, std::abs(s_simd[i] - s_std[i]));
    max_abs_err_c = std::max(max_abs_err_c, std::abs(c_simd[i] - c_std[i]));
    max_rel_err_s = std::max(max_rel_err_s, rel(s_simd[i], s_std[i]));
    max_rel_err_c = std::max(max_rel_err_c, rel(c_simd[i], c_std[i]));
  }

  std::printf("W=%zu  n=%zu  accuracy:\n", W, n);
  std::printf("  sin: max abs err = %.3e   max rel err = %.3e\n", max_abs_err_s, max_rel_err_s);
  std::printf("  cos: max abs err = %.3e   max rel err = %.3e\n", max_abs_err_c, max_rel_err_c);
}

template<size_t W>
static void speed_check(size_t n, size_t iters)
{
  std::vector<double> x(n), s(n), c(n);
  // mildly structured input (avoid denorm storms; exercise range around multiples of pi)
  for (size_t i = 0; i < n; ++i) x[i] = (i % 1000) * 0.001 * 100.0 - 50.0;

  // warmup
  simd_sincos_array<W>(x.data(), s.data(), c.data(), n);

  auto t0 = clk::now();
  for (size_t k = 0; k < iters; ++k) simd_sincos_array<W>(x.data(), s.data(), c.data(), n);
  auto t1 = clk::now();

  auto t2 = clk::now();
  for (size_t k = 0; k < iters; ++k) scalar_sincos_array(x.data(), s.data(), c.data(), n);
  auto t3 = clk::now();

  double tsimd = std::chrono::duration_cast<sec>(t1 - t0).count();
  double tscalar = std::chrono::duration_cast<sec>(t3 - t2).count();

  // report ns/value
  double ops = double(n) * double(iters);
  std::printf("W=%zu  n=%zu  iters=%zu\n", W, n, iters);
  std::printf("  SIMD:   %.3f ns/value\n", 1e9 * tsimd / ops);
  std::printf("  scalar: %.3f ns/value\n", 1e9 * tscalar / ops);
  std::printf("  speedup: %.2fx\n", tscalar / tsimd);
}

template<size_t W>
static void run_all(size_t n, size_t iters, double rmin, double rmax)
{
  accuracy_check<W>(n, rmin, rmax);
  speed_check<W>(n, iters);
}

int main(int argc, char** argv)
{
  // defaults
  size_t n = 1 << 26;       // 65,536 inputs
  size_t iters = 1;       // repeat to amplify timing
  int reqW = 8;             // default SIMD width
  double rmin = -1.0e3, rmax = 1.0e3;

  if (argc > 1) reqW = std::stoi(argv[1]);
  if (argc > 2) n = size_t(std::stoull(argv[2]));
  if (argc > 3) iters = size_t(std::stoull(argv[3]));
  if (argc > 5) { rmin = std::stod(argv[4]); rmax = std::stod(argv[5]); }

  switch (reqW) {
    case 1: run_all<1>(n, iters, rmin, rmax); break;
    case 2: run_all<2>(n, iters, rmin, rmax); break;
    case 4: run_all<4>(n, iters, rmin, rmax); break;
    case 8: run_all<8>(n, iters, rmin, rmax); break;
    case 16: run_all<16>(n, iters, rmin, rmax); break;
    case 32: run_all<32>(n, iters, rmin, rmax); break;
    default:
      std::fprintf(stderr, "Unsupported width %d (use 1, 2, or 4)\n", reqW);
      return 2;
  }
  return 0;
}
