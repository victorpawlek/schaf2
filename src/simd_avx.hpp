#ifndef SIMD_AVX_HPP
#define SIMD_AVX_HPP

#include <immintrin.h>
#include <cstdint>
#include <cmath>


/*
  implementation of SIMDs for Intel-CPUs with AVX support:
  https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
 */


namespace ASC_HPC
{

  //mask for 2 values
  template<>
  class SIMD<mask64,2>
  {
    __m128i m_mask;
  public:

    SIMD (__m128i mask) : m_mask(mask) { };
    SIMD (__m128d mask) : m_mask(_mm_castpd_si128(mask)) { ; }
    SIMD(mask64 m0, mask64 m1)
      : m_mask(_mm_set_epi64x(m1 ? ~0LL : 0LL,   // hi
                              m0 ? ~0LL : 0LL))  // lo
    {}
    SIMD(SIMD<mask64,1> m0, SIMD<mask64,1> m1) : SIMD( mask64(m0[0]), mask64(m1[0]) ) {}
    auto val() const { return m_mask; }
    mask64 operator[](size_t i) const { return ( (int64_t*)&m_mask)[i] != 0; }
    
    SIMD<mask64, 1> lo() const { return SIMD<mask64,1>((*this)[0]); }
    SIMD<mask64, 1> hi() const { return SIMD<mask64,1>((*this)[1]); }
  };

  // mask for 4 values
  template<>
  class SIMD<mask64,4>
  {
    __m256i m_mask;
  public:

    SIMD (__m256i mask) : m_mask(mask) { };
    SIMD (__m256d mask) : m_mask(_mm256_castpd_si256(mask)) { ; }
    auto val() const { return m_mask; }
    mask64 operator[](size_t i) const { return ( (int64_t*)&m_mask)[i] != 0; }
    SIMD(SIMD<mask64,2> lo, SIMD<mask64,2> hi): m_mask( _mm256_set_m128i(hi.val(), lo.val()) ) {}
    
    SIMD<mask64, 2> lo() const { return SIMD<mask64,2>((*this)[0], (*this)[1]); }
    SIMD<mask64, 2> hi() const { return SIMD<mask64,2>((*this)[2], (*this)[3]); }
  };


  // SIMD class with 2 values
  template<>
  class SIMD<double,2>
  {
    __m128d m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD (double val) : m_val{_mm_set1_pd(val)} {};
    SIMD (__m128d val) : m_val{val} {};
    SIMD (double v0, double v1) : m_val{_mm_set_pd(v1,v0)} {  }
    SIMD (SIMD<double,1> v0, SIMD<double,1> v1) : SIMD(v0[0], v1[0]) { }
    SIMD (std::array<double,2> a) : SIMD(a[0],a[1]) { }
    SIMD (double const * p) { m_val = _mm_loadu_pd(p); }
    SIMD (double const * p, SIMD<mask64,2> mask) { m_val = _mm_maskload_pd(p, mask.val()); }
    
    static constexpr int size() { return 2; }
    auto val() const { return m_val; }
    const double * ptr() const { return (double*)&m_val; }
    SIMD<double, 1> lo() const { return SIMD<double,1>((*this)[0]); }
    SIMD<double, 1> hi() const { return SIMD<double,1>((*this)[1]); }

    // better:
    // SIMD<double, 2> lo() const { return _mm256_extractf128_pd(m_val, 0); }
    // SIMD<double, 2> hi() const { return _mm256_extractf128_pd(m_val, 1); }
    double operator[](size_t i) const { return ((double*)&m_val)[i]; }

    void store (double * p) const { _mm_storeu_pd(p, m_val); }
    void store (double * p, SIMD<mask64, 2> mask) const { _mm_maskstore_pd(p, mask.val(), m_val); }
  };

  // SIMD class for 4 values
  template<>
  class SIMD<double,4>
  {
    __m256d m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD (double val) : m_val{_mm256_set1_pd(val)} {};
    SIMD (__m256d val) : m_val{val} {};
    SIMD (double v0, double v1, double v2, double v3) : m_val{_mm256_set_pd(v3,v2,v1,v0)} {  }
    SIMD (SIMD<double,2> v0, SIMD<double,2> v1) : SIMD(v0[0], v0[1], v1[0], v1[1]) { }  // better with _mm256_set_m128d
    SIMD (std::array<double,4> a) : SIMD(a[0],a[1],a[2],a[3]) { }
    SIMD (double const * p) { m_val = _mm256_loadu_pd(p); }
    SIMD (double const * p, SIMD<mask64,4> mask) { m_val = _mm256_maskload_pd(p, mask.val()); }
    
    static constexpr int size() { return 4; }
    auto val() const { return m_val; }
    const double * ptr() const { return (double*)&m_val; }
    SIMD<double, 2> lo() const { return SIMD<double,2>((*this)[0], (*this)[1]); }
    SIMD<double, 2> hi() const { return SIMD<double,2>((*this)[2], (*this)[3]); }

    // better:
    // SIMD<double, 2> lo() const { return _mm256_extractf128_pd(m_val, 0); }
    // SIMD<double, 2> hi() const { return _mm256_extractf128_pd(m_val, 1); }
    double operator[](size_t i) const { return ((double*)&m_val)[i]; }

    void store (double * p) const { _mm256_storeu_pd(p, m_val); }
    void store (double * p, SIMD<mask64,4> mask) const { _mm256_maskstore_pd(p, mask.val(), m_val); }
  };


  template<>
  class SIMD<int64_t,2>
  {
    __m128i m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(int64_t val) : m_val{_mm_set1_epi64x(val)} {};
    SIMD(__m128i val) : m_val{val} {};
    SIMD (int64_t v0, int64_t v1) : m_val{_mm_set_epi64x(v1,v0) } { } 
    SIMD (SIMD<int64_t,1> v0, SIMD<int64_t,1> v1) : SIMD(v0[0], v1[0]) { }  // can do better !
    SIMD (std::array<int64_t,2> a) : SIMD(a[0],a[1]) { }
    SIMD(const int64_t* p) {
      m_val = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    }
    SIMD (int64_t const * p, SIMD<mask64,2> mask) { m_val = _mm_maskload_epi64(p, mask.val()); }
    
    static constexpr int size() { return 2; }
    auto val() const { return m_val; }
    // const int64_t * ptr() const { return (int64_t*)&val; }
    SIMD<int64_t, 1> lo() const { return _mm_extract_epi64(m_val, 0); }
    SIMD<int64_t, 1> hi() const { return _mm_extract_epi64(m_val, 1); }
    int64_t operator[](size_t i) const { return ((int64_t*)&m_val)[i]; }
  }; 
  
  template<>
  class SIMD<int64_t,4>
  {
    __m256i m_val;
  public:
    SIMD () = default;
    SIMD (const SIMD &) = default;
    SIMD(int64_t val) : m_val{_mm256_set1_epi64x(val)} {};
    SIMD(__m256i val) : m_val{val} {};
    SIMD (int64_t v0, int64_t v1, int64_t v2, int64_t v3) : m_val{_mm256_set_epi64x(v3,v2,v1,v0) } { } 
    SIMD (SIMD<int64_t,2> v0, SIMD<int64_t,2> v1) : SIMD(v0[0], v0[1], v1[0], v1[1]) { }  // can do better !
    SIMD (std::array<int64_t,4> a) : SIMD(a[0],a[1],a[2],a[3]) { }
    SIMD(const int64_t* p) {
      m_val = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    }
    SIMD (int64_t const * p, SIMD<mask64,4> mask) { m_val = _mm256_maskload_epi64(p, mask.val()); }
    
    static constexpr int size() { return 4; }
    auto val() const { return m_val; }
    const int64_t * ptr() const { return (int64_t*)&m_val; }
    SIMD<int64_t, 2> lo() const { return _mm256_extracti128_si256(m_val, 0); }
    SIMD<int64_t, 2> hi() const { return _mm256_extracti128_si256(m_val, 1); }
    int64_t operator[](size_t i) const { return ((int64_t*)&m_val)[i]; }
  };


  template <int64_t first>
  class IndexSequence<int64_t, 4, first> : public SIMD<int64_t,4>
  {
  public:
    IndexSequence()
      : SIMD<int64_t,4> (first, first+1, first+2, first+3) { }
  };
  


  //arithmetic for 4  
  inline auto operator+ (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_add_pd(a.val(), b.val())); }
  inline auto operator- (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_sub_pd(a.val(), b.val())); }
  
  inline auto operator* (SIMD<double,4> a, SIMD<double,4> b) { return SIMD<double,4> (_mm256_mul_pd(a.val(), b.val())); }
  inline auto operator* (double a, SIMD<double,4> b) { return SIMD<double,4>(a)*b; }
  
  // arithmetic for 2
  inline auto operator+ (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (_mm_add_pd(a.val(), b.val())); }
  inline auto operator- (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (_mm_sub_pd(a.val(), b.val())); }
  
  inline auto operator* (SIMD<double,2> a, SIMD<double,2> b) { return SIMD<double,2> (_mm_mul_pd(a.val(), b.val())); }
  inline auto operator* (double a, SIMD<double,2> b) { return SIMD<double,2>(a)*b; }

  //?
  #ifdef __FMA__
    inline SIMD<double,4> fma (SIMD<double,4> a, SIMD<double,4> b, SIMD<double,4> c)
    { return _mm256_fmadd_pd (a.val(), b.val(), c.val()); }
  #endif

  inline SIMD<mask64,4> operator>= (SIMD<int64_t,4> a , SIMD<int64_t,4> b)
  { // there is no a>=b, so we return !(b>a)
    return  _mm256_xor_si256(_mm256_cmpgt_epi64(b.val(),a.val()),_mm256_set1_epi32(-1)); }
  
  inline auto operator>= (SIMD<double,4> a, SIMD<double,4> b)
  { return SIMD<mask64,4>(_mm256_cmp_pd (a.val(), b.val(), _CMP_GE_OQ)); }


  inline SIMD<double,4> round(const SIMD<double,4>& x) {
    __m256d r = _mm256_round_pd(x.val(), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return SIMD<double,4>(r);
  }

  inline SIMD<double,2> round(const SIMD<double,2>& x) {
    __m128d r = _mm_round_pd(x.val(), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return SIMD<double,2>(r);
  }

  // inline SIMD<double,1> round(const SIMD<double,1>& x) {
  //   return SIMD<double,1>( std::round(x.val()) );
  // }

  // inline SIMD<double,1> lround(const SIMD<double,1>& x) {
  //   return SIMD<double,1>( std::lround(x.val()) );
  // }

  // inline SIMD<int64_t,1> lround(const SIMD<int64_t,1>& x) {
  //   return SIMD<int64_t,1>( std::lround(x.val()) );
  // }
  

  
}

#endif
