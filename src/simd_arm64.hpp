#ifndef SIMD_ARM64_HPP
#define SIMD_ARM64_HPP

#include "arm_neon.h"

/*
  implementation of SIMDs for ARM-Neon CPUs:
  https://arm-software.github.io/acle/neon_intrinsics/advsimd.html

  // neon coding:
  https://developer.arm.com/documentation/102159/0400
*/


namespace ASC_HPC
{

  template<>
  class SIMD<mask64,2>
  {
    int64x2_t m_val;
  public:
    SIMD (int64x2_t val) : m_val(val) { };
    SIMD (SIMD<mask64,1> v0, SIMD<mask64,1> v1)
      : m_val{vcombine_s64(int64x1_t{v0.val().val()}, int64x1_t{v1.val().val()})} { } 

    auto val() const { return m_val; }
    mask64 operator[](size_t i) const { return ( (int64_t*)&m_val)[i] != 0; }

    SIMD<mask64, 1> lo() const { return SIMD<mask64,1>((*this)[0]); }
    SIMD<mask64, 1> hi() const { return SIMD<mask64,1>((*this)[1]); }
    const mask64 * ptr() const { return (mask64*)&m_val; }
  };




  
  template<>
  class SIMD<double,2>
  {
    float64x2_t m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD (double val) : m_val{val,val} { }
    SIMD (float64x2_t val) : m_val(val) { }
    SIMD (double v0, double v1) : m_val{vcombine_f64(float64x1_t{v0}, float64x1_t{v1})} { }
    SIMD (SIMD<double,1> v0, SIMD<double,1> v1) : SIMD(v0.val(), v1.val()) { }
    SIMD (std::array<double, 2> arr) : m_val{arr[0], arr[1]} { }

    SIMD (double const * p) : m_val{vld1q_f64(p)} { }
    SIMD (double const * p, SIMD<mask64,2> mask)
      {
	m_val[0] = mask[0] ? p[0] : 0;
	m_val[1] = mask[1] ? p[1] : 0;
      }

    static constexpr int size() { return 2; }    
    auto val() const { return m_val; }
    const double * ptr() const { return (double*)&m_val; }

    auto lo() const { return SIMD<double,1> (m_val[0]); }
    auto hi() const { return SIMD<double,1> (m_val[1]); }
    double operator[] (int i) const { return m_val[i]; }

    void store (double * p) const
    {
      vst1q_f64(p, m_val);
    }
    
    void store (double * p, SIMD<mask64,2> mask) const
    {
      if (mask[0]) p[0] = m_val[0];
      if (mask[1]) p[1] = m_val[1];
    }
  };

  inline auto operator+ (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (a.val()+b.val()); }
  inline auto operator- (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (a.val()-b.val()); }
  
  inline auto operator* (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (a.val()*b.val()); }
  inline auto operator* (double a, SIMD<double,2> b) { return SIMD<double,2> (a*b.val()); }    
  
  // a*b+c
  inline SIMD<double,2> fma (SIMD<double,2> a, SIMD<double,2> b, SIMD<double,2> c) 
  { return vmlaq_f64(c.val(), a.val(), b.val()); }



  inline SIMD<double,2> select (SIMD<mask64,2> mask, SIMD<double,2> b, SIMD<double,2> c)
  { return vbslq_f64(mask.val(), b.val(), c.val()); }
  
  inline SIMD<double,2> hSum (SIMD<double,2> a, SIMD<double,2> b)
  { return vpaddq_f64(a.val(), b.val()); }

  inline SIMD<mask64,2> operator>= (SIMD<double,2> a, SIMD<double,2> b)
  { return SIMD<mask64,2>(vreinterpretq_s64_u64(vcgeq_f64(a.val(), b.val()))); }

  inline SIMD<mask64,2> operator< (SIMD<double,2> a, SIMD<double,2> b)
  { return SIMD<mask64,2>(vreinterpretq_s64_u64(vcltq_f64(a.val(), b.val()))); }

  inline SIMD<mask64,2> operator<= (SIMD<double,2> a, SIMD<double,2> b)
  { return SIMD<mask64,2>(vreinterpretq_s64_u64(vcleq_f64(a.val(), b.val()))); }

  inline SIMD<mask64,2> operator> (SIMD<double,2> a, SIMD<double,2> b)
  { return SIMD<mask64,2>(vreinterpretq_s64_u64(vcgtq_f64(a.val(), b.val()))); }

  inline SIMD<mask64,2> operator== (SIMD<double,2> a, SIMD<double,2> b)
  { return SIMD<mask64,2>(vreinterpretq_s64_u64(vceqq_f64(a.val(), b.val()))); }

  inline SIMD<mask64,2> operator!= (SIMD<double,2> a, SIMD<double,2> b)
  {
    const uint64x2_t eq = vceqq_f64(a.val(), b.val());
    const uint64x2_t mask_all = vdupq_n_u64(~uint64_t(0));
    return SIMD<mask64,2>(vreinterpretq_s64_u64(veorq_u64(eq, mask_all)));
  }

  inline void transpose (SIMD<double,4> a0, SIMD<double,4> a1, SIMD<double,4> a2, SIMD<double,4> a3,
                         SIMD<double,4> &b0, SIMD<double,4> &b1, SIMD<double,4> &b2, SIMD<double,4> &b3)
  {
    // interleave row pairs and build the columns as SIMD<double,4> composed from 128-bit halves
    const float64x2_t a0_lo = a0.lo().val();
    const float64x2_t a0_hi = a0.hi().val();
    const float64x2_t a1_lo = a1.lo().val();
    const float64x2_t a1_hi = a1.hi().val();
    const float64x2_t a2_lo = a2.lo().val();
    const float64x2_t a2_hi = a2.hi().val();
    const float64x2_t a3_lo = a3.lo().val();
    const float64x2_t a3_hi = a3.hi().val();

    const float64x2_t r01_lo_0 = vzip1q_f64(a0_lo, a1_lo);
    const float64x2_t r01_lo_1 = vzip2q_f64(a0_lo, a1_lo);
    const float64x2_t r23_lo_0 = vzip1q_f64(a2_lo, a3_lo);
    const float64x2_t r23_lo_1 = vzip2q_f64(a2_lo, a3_lo);

    const float64x2_t r01_hi_0 = vzip1q_f64(a0_hi, a1_hi);
    const float64x2_t r01_hi_1 = vzip2q_f64(a0_hi, a1_hi);
    const float64x2_t r23_hi_0 = vzip1q_f64(a2_hi, a3_hi);
    const float64x2_t r23_hi_1 = vzip2q_f64(a2_hi, a3_hi);

    b0 = SIMD<double,4>(SIMD<double,2>(r01_lo_0), SIMD<double,2>(r23_lo_0));
    b1 = SIMD<double,4>(SIMD<double,2>(r01_lo_1), SIMD<double,2>(r23_lo_1));
    b2 = SIMD<double,4>(SIMD<double,2>(r01_hi_0), SIMD<double,2>(r23_hi_0));
    b3 = SIMD<double,4>(SIMD<double,2>(r01_hi_1), SIMD<double,2>(r23_hi_1));
  }
  
}

#endif
