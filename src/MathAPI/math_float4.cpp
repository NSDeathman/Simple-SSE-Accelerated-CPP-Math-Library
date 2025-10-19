// Description: 4-dimensional vector class with comprehensive 
//              mathematical operations and SSE optimization
//              Supports both 4D vectors and homogeneous coordinates
// Author: NSDeathman, DeepSeek

#include "math_float4.h"

namespace Math
{
    // ============================================================================
    // Constructors Implementation
    // ============================================================================

    float4::float4() noexcept : simd_(_mm_setzero_ps()) {}

    float4::float4(float x, float y, float z, float w) noexcept : simd_(_mm_set_ps(w, z, y, x)) {}

    float4::float4(float scalar) noexcept : simd_(_mm_set1_ps(scalar)) {}

    float4::float4(const float2& vec, float z, float w) noexcept
        : simd_(_mm_set_ps(w, z, vec.y, vec.x)) {}

    float4::float4(const float3& vec, float w) noexcept
        : simd_(_mm_set_ps(w, vec.z, vec.y, vec.x)) {}

    float4::float4(const float* data) noexcept : simd_(_mm_loadu_ps(data)) {}

    float4::float4(__m128 simd_val) noexcept : simd_(simd_val) {}

#if defined(MATH_SUPPORT_D3DX)
    float4::float4(const D3DXVECTOR4& vec) noexcept : simd_(_mm_loadu_ps(&vec.x)) {}

    float4::float4(const D3DXVECTOR3& vec, float w) noexcept
        : simd_(_mm_set_ps(w, vec.z, vec.y, vec.x)) {}

    float4::float4(const D3DXVECTOR2& vec, float z, float w) noexcept
        : simd_(_mm_set_ps(w, z, vec.y, vec.x)) {}

    float4::float4(D3DCOLOR color) noexcept {
        float r = static_cast<float>((color >> 16) & 0xFF) / 255.0f;
        float g = static_cast<float>((color >> 8) & 0xFF) / 255.0f;
        float b = static_cast<float>(color & 0xFF) / 255.0f;
        float a = static_cast<float>((color >> 24) & 0xFF) / 255.0f;
        simd_ = _mm_set_ps(a, b, g, r);
    }
#endif

    // ============================================================================
    // Assignment Operators Implementation
    // ============================================================================

    float4& float4::operator=(float scalar) noexcept {
        simd_ = _mm_set1_ps(scalar);
        return *this;
    }

    float4& float4::operator=(const float3& xyz) noexcept {
        // ��������� w ���������, ��������� xyz
        __m128 xyz_vec = _mm_set_ps(0.0f, xyz.z, xyz.y, xyz.x);
        simd_ = _mm_blend_ps(xyz_vec, simd_, 0x8); // ��������� w �� �������� ��������
        return *this;
    }

#if defined(MATH_SUPPORT_D3DX)
    float4& float4::operator=(const D3DXVECTOR4& vec) noexcept {
        simd_ = _mm_loadu_ps(&vec.x);
        return *this;
    }

    float4& float4::operator=(D3DCOLOR color) noexcept {
        float r = static_cast<float>((color >> 16) & 0xFF) / 255.0f;
        float g = static_cast<float>((color >> 8) & 0xFF) / 255.0f;
        float b = static_cast<float>(color & 0xFF) / 255.0f;
        float a = static_cast<float>((color >> 24) & 0xFF) / 255.0f;
        simd_ = _mm_set_ps(a, b, g, r);
        return *this;
    }
#endif

    // ============================================================================
    // Compound Assignment Operators Implementation
    // ============================================================================

    float4& float4::operator+=(const float4& rhs) noexcept {
        simd_ = _mm_add_ps(simd_, rhs.simd_);
        return *this;
    }

    float4& float4::operator-=(const float4& rhs) noexcept {
        simd_ = _mm_sub_ps(simd_, rhs.simd_);
        return *this;
    }

    float4& float4::operator*=(const float4& rhs) noexcept {
        simd_ = _mm_mul_ps(simd_, rhs.simd_);
        return *this;
    }

    float4& float4::operator/=(const float4& rhs) noexcept {
        simd_ = _mm_div_ps(simd_, rhs.simd_);
        return *this;
    }

    float4& float4::operator*=(float scalar) noexcept {
        __m128 scalar_vec = _mm_set1_ps(scalar);
        simd_ = _mm_mul_ps(simd_, scalar_vec);
        return *this;
    }

    float4& float4::operator/=(float scalar) noexcept {
        __m128 inv_scalar = _mm_set1_ps(1.0f / scalar);
        simd_ = _mm_mul_ps(simd_, inv_scalar);
        return *this;
    }

    // ============================================================================
    // Unary Operators Implementation
    // ============================================================================

    float4 float4::operator+() const noexcept {
        return *this;
    }

    float4 float4::operator-() const noexcept {
        __m128 neg = _mm_set1_ps(-1.0f);
        return float4(_mm_mul_ps(simd_, neg));
    }

    // ============================================================================
    // Access Operators Implementation
    // ============================================================================

    float& float4::operator[](int index) noexcept {
        return (&x)[index];
    }

    const float& float4::operator[](int index) const noexcept {
        return (&x)[index];
    }

    // ============================================================================
    // Conversion Operators Implementation
    // ============================================================================

    float4::operator const float* () const noexcept { return &x; }

    float4::operator float* () noexcept { return &x; }

    float4::operator __m128() const noexcept { return simd_; }

#if defined(MATH_SUPPORT_D3DX)
    float4::operator D3DXVECTOR4() const noexcept {
        D3DXVECTOR4 result;
        _mm_storeu_ps(&result.x, simd_);
        return result;
    }
#endif

    // ============================================================================
    // Static Constructors Implementation
    // ============================================================================

    float4 float4::zero() noexcept {
        return float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float4 float4::one() noexcept {
        return float4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    float4 float4::unit_x() noexcept {
        return float4(1.0f, 0.0f, 0.0f, 0.0f);
    }

    float4 float4::unit_y() noexcept {
        return float4(0.0f, 1.0f, 0.0f, 0.0f);
    }

    float4 float4::unit_z() noexcept {
        return float4(0.0f, 0.0f, 1.0f, 0.0f);
    }

    float4 float4::unit_w() noexcept {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    float4 float4::from_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) noexcept {
        return float4(
            static_cast<float>(r) / 255.0f,
            static_cast<float>(g) / 255.0f,
            static_cast<float>(b) / 255.0f,
            static_cast<float>(a) / 255.0f
        );
    }

    float4 float4::from_color(float r, float g, float b, float a) noexcept {
        return float4(r, g, b, a);
    }

    // ============================================================================
    // Mathematical Functions Implementation
    // ============================================================================

    float float4::length() const noexcept {
        __m128 squared = _mm_mul_ps(simd_, simd_);
        __m128 sum = _mm_hadd_ps(squared, squared);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(_mm_sqrt_ss(sum));
    }

    float float4::length_sq() const noexcept {
        __m128 squared = _mm_mul_ps(simd_, simd_);
        __m128 sum = _mm_hadd_ps(squared, squared);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }

    float4 float4::normalize() const noexcept {
        __m128 squared = _mm_mul_ps(simd_, simd_);
        __m128 sum = _mm_hadd_ps(squared, squared);
        sum = _mm_hadd_ps(sum, sum);

        __m128 len_vec = _mm_sqrt_ps(sum);
        __m128 mask = _mm_cmpgt_ps(len_vec, _mm_set1_ps(Constants::Constants<float>::Epsilon));
        __m128 inv_len = _mm_div_ps(_mm_set1_ps(1.0f), len_vec);
        inv_len = _mm_and_ps(inv_len, mask); // ���� ����� 0, �� inv_len = 0

        return float4(_mm_mul_ps(simd_, inv_len));
    }

    float float4::dot(const float4& other) const noexcept {
        return float4::dot(*this, other);
    }

    float float4::dot3(const float4& other) const noexcept {
        return float4::dot3(*this, other);
    }

    float4 float4::cross(const float4& other) const noexcept {
        return float4::cross(*this, other);
    }

    float float4::distance(const float4& other) const noexcept {
        __m128 diff = _mm_sub_ps(simd_, other.simd_);
        __m128 squared = _mm_mul_ps(diff, diff);
        __m128 sum = _mm_hadd_ps(squared, squared);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(_mm_sqrt_ss(sum));
    }

    float float4::distance_sq(const float4& other) const noexcept {
        return (*this - other).length_sq();
    }

    // ============================================================================
    // HLSL-like Functions Implementation
    // ============================================================================

    float4 float4::abs() const noexcept {
        __m128 mask = _mm_set1_ps(-0.0f); // -0.0f = 0x80000000
        return float4(_mm_andnot_ps(mask, simd_));
    }

    float4 float4::sign() const noexcept {
        __m128 zero = _mm_setzero_ps();
        __m128 one = _mm_set1_ps(1.0f);
        __m128 neg_one = _mm_set1_ps(-1.0f);

        __m128 gt_zero = _mm_cmpgt_ps(simd_, zero);
        __m128 lt_zero = _mm_cmplt_ps(simd_, zero);

        __m128 result = _mm_and_ps(gt_zero, one);
        result = _mm_or_ps(result, _mm_and_ps(lt_zero, neg_one));

        return float4(result);
    }

    float4 float4::floor() const noexcept {
#ifdef __SSE4_1__
        return float4(_mm_floor_ps(simd_));
#else
        alignas(16) float temp[4];
        _mm_store_ps(temp, simd_);
        return float4(std::floor(temp[0]), std::floor(temp[1]), std::floor(temp[2]), std::floor(temp[3]));
#endif
    }

    float4 float4::ceil() const noexcept {
#ifdef __SSE4_1__
        return float4(_mm_ceil_ps(simd_));
#else
        alignas(16) float temp[4];
        _mm_store_ps(temp, simd_);
        return float4(std::ceil(temp[0]), std::ceil(temp[1]), std::ceil(temp[2]), std::ceil(temp[3]));
#endif
    }

    float4 float4::round() const noexcept {
#ifdef __SSE4_1__
        return float4(_mm_round_ps(simd_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
        alignas(16) float temp[4];
        _mm_store_ps(temp, simd_);
        return float4(std::round(temp[0]), std::round(temp[1]), std::round(temp[2]), std::round(temp[3]));
#endif
    }

    float4 float4::frac() const noexcept {
        alignas(16) float temp[4];
        _mm_store_ps(temp, simd_);
        return float4(
            temp[0] - std::floor(temp[0]),
            temp[1] - std::floor(temp[1]),
            temp[2] - std::floor(temp[2]),
            temp[3] - std::floor(temp[3])
        );
    }

    float4 float4::saturate() const noexcept {
        return float4::saturate(*this);
    }

    float4 float4::step(float edge) const noexcept {
        __m128 edge_vec = _mm_set1_ps(edge);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 zero = _mm_setzero_ps();

        __m128 cmp = _mm_cmpge_ps(simd_, edge_vec);
        return float4(_mm_or_ps(_mm_and_ps(cmp, one), _mm_andnot_ps(cmp, zero)));
    }

    // ============================================================================
    // Color Operations Implementation
    // ============================================================================

    float float4::luminance() const noexcept {
        __m128 weights = _mm_set_ps(0.0f, 0.0722f, 0.7152f, 0.2126f);
        __m128 mul = _mm_mul_ps(simd_, weights);
        __m128 sum = _mm_hadd_ps(mul, mul);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }

    float float4::brightness() const noexcept {
        __m128 sum = _mm_hadd_ps(simd_, simd_);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(_mm_mul_ss(sum, _mm_set1_ps(1.0f / 3.0f)));
    }

    float4 float4::premultiply_alpha() const noexcept {
        __m128 alpha = _mm_shuffle_ps(simd_, simd_, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 result = _mm_mul_ps(simd_, alpha);
        result = _mm_blend_ps(result, simd_, 0x8);
        return float4(result);
    }

    float4 float4::unpremultiply_alpha() const noexcept {
        // ��������� alpha �� 0
        __m128 alpha = _mm_shuffle_ps(simd_, simd_, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 zero = _mm_setzero_ps();
        __m128 alpha_is_zero = _mm_cmpeq_ps(alpha, zero);

        // ���� alpha = 0, ���������� ������������ ������
        if (_mm_movemask_ps(alpha_is_zero) != 0) {
            return *this;
        }

        // ����� ����� RGB �� alpha
        __m128 inv_alpha = _mm_div_ps(_mm_set1_ps(1.0f), alpha);

        // �������� ��� ���������� �� inverse alpha
        __m128 result = _mm_mul_ps(simd_, inv_alpha);

        // �� ��������������� ������������ alpha (�� ������ alpha �� ���� ����)
        result = _mm_blend_ps(result, simd_, 0x8); // ����� 1000 - ��������� ������������ alpha

        return float4(result);
    }

    float4 float4::grayscale() const noexcept {
        float lum = luminance();
        return float4(lum, lum, lum, w);
    }

    // ============================================================================
    // Geometric Operations Implementation
    // ============================================================================

    float3 float4::project() const noexcept {
        __m128 w_vec = _mm_shuffle_ps(simd_, simd_, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 mask = _mm_cmpneq_ps(w_vec, _mm_setzero_ps());
        __m128 inv_w = _mm_div_ps(_mm_set1_ps(1.0f), w_vec);
        inv_w = _mm_and_ps(inv_w, mask);

        __m128 projected = _mm_mul_ps(simd_, inv_w);
        return float3(_mm_cvtss_f32(projected),
            _mm_cvtss_f32(_mm_shuffle_ps(projected, projected, _MM_SHUFFLE(1, 1, 1, 1))),
            _mm_cvtss_f32(_mm_shuffle_ps(projected, projected, _MM_SHUFFLE(2, 2, 2, 2))));
    }

    float4 float4::to_homogeneous() const noexcept {
        return float4(x, y, z, 1.0f);
    }

    // ============================================================================
    // Static Mathematical Functions Implementation
    // ============================================================================

    float float4::dot(const float4& a, const float4& b) noexcept {
        __m128 mul = _mm_mul_ps(a.simd_, b.simd_);
        __m128 sum = _mm_hadd_ps(mul, mul);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }

    float float4::dot3(const float4& a, const float4& b) noexcept {
        // ������� � �������� ���������� ��� ������� �����
        __m128 a3 = a.simd_;
        __m128 b3 = b.simd_;

        // �������� ��� ����������
        __m128 mul = _mm_mul_ps(a3, b3);

        // ��������� ������ x, y, z (���������� w)
        // ���������� �������������� ��������
        __m128 sum_xy = _mm_hadd_ps(mul, mul);        // x+y, z+w, x+y, z+w
        __m128 sum_xyz = _mm_hadd_ps(sum_xy, sum_xy); // x+y+z+w, x+y+z+w, x+y+z+w, x+y+z+w

        // ������ �������� w ��������� ����� �������� ������ x+y+z
        __m128 w_component = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(3, 3, 3, 3)); // w, w, w, w
        __m128 result = _mm_sub_ps(sum_xyz, w_component);

        return _mm_cvtss_f32(result);
    }

    float4 float4::cross(const float4& a, const float4& b) noexcept {
        __m128 a_vec = a.simd_;
        __m128 b_vec = b.simd_;

        // a.y * b.z - a.z * b.y
        // a.z * b.x - a.x * b.z  
        // a.x * b.y - a.y * b.x
        // w = 0

        __m128 a_y = _mm_shuffle_ps(a_vec, a_vec, _MM_SHUFFLE(3, 3, 0, 1)); // a.y, a.y, a.y, a.y
        __m128 a_z = _mm_shuffle_ps(a_vec, a_vec, _MM_SHUFFLE(3, 3, 1, 2)); // a.z, a.z, a.z, a.z  
        __m128 a_x = _mm_shuffle_ps(a_vec, a_vec, _MM_SHUFFLE(3, 3, 2, 0)); // a.x, a.x, a.x, a.x

        __m128 b_y = _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(3, 3, 0, 1)); // b.y, b.y, b.y, b.y
        __m128 b_z = _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(3, 3, 1, 2)); // b.z, b.z, b.z, b.z
        __m128 b_x = _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(3, 3, 2, 0)); // b.x, b.x, b.x, b.x

        __m128 result_x = _mm_sub_ps(_mm_mul_ps(a_y, b_z), _mm_mul_ps(a_z, b_y));
        __m128 result_y = _mm_sub_ps(_mm_mul_ps(a_z, b_x), _mm_mul_ps(a_x, b_z));
        __m128 result_z = _mm_sub_ps(_mm_mul_ps(a_x, b_y), _mm_mul_ps(a_y, b_x));

        __m128 result = _mm_set_ps(0.0f,
            _mm_cvtss_f32(result_z),
            _mm_cvtss_f32(result_y),
            _mm_cvtss_f32(result_x));

        return float4(result);
    }

    float4 float4::lerp(const float4& a, const float4& b, float t) noexcept {
        __m128 t_vec = _mm_set1_ps(t);
        __m128 one_minus_t = _mm_set1_ps(1.0f - t);

        __m128 part1 = _mm_mul_ps(a.simd_, one_minus_t);
        __m128 part2 = _mm_mul_ps(b.simd_, t_vec);

        return float4(_mm_add_ps(part1, part2));
    }

    float4 float4::min(const float4& a, const float4& b) noexcept {
        return float4(_mm_min_ps(a.simd_, b.simd_));
    }

    float4 float4::max(const float4& a, const float4& b) noexcept {
        return float4(_mm_max_ps(a.simd_, b.simd_));
    }

    float4 float4::saturate(const float4& vec) noexcept {
        __m128 zero = _mm_setzero_ps();
        __m128 one = _mm_set1_ps(1.0f);
        __m128 result = _mm_max_ps(vec.simd_, zero);
        result = _mm_min_ps(result, one);
        return float4(result);
    }

    // ============================================================================
    // Swizzle Operations Implementation
    // ============================================================================

    float2 float4::xy() const noexcept { return float2(x, y); }
    float2 float4::xz() const noexcept { return float2(x, z); }
    float2 float4::xw() const noexcept { return float2(x, w); }
    float2 float4::yz() const noexcept { return float2(y, z); }
    float2 float4::yw() const noexcept { return float2(y, w); }
    float2 float4::zw() const noexcept { return float2(z, w); }

    float3 float4::xyz() const noexcept { return float3(x, y, z); }
    float3 float4::xyw() const noexcept { return float3(x, y, w); }
    float3 float4::xzw() const noexcept { return float3(x, z, w); }
    float3 float4::yzw() const noexcept { return float3(y, z, w); }

    float4 float4::yxzw() const noexcept { return float4(y, x, z, w); }
    float4 float4::zxyw() const noexcept { return float4(z, x, y, w); }
    float4 float4::zyxw() const noexcept { return float4(z, y, x, w); }
    float4 float4::wzyx() const noexcept { return float4(w, z, y, x); }

    // Color swizzles
    float float4::r() const noexcept { return x; }
    float float4::g() const noexcept { return y; }
    float float4::b() const noexcept { return z; }
    float float4::a() const noexcept { return w; }
    float2 float4::rg() const noexcept { return float2(x, y); }
    float2 float4::rb() const noexcept { return float2(x, z); }
    float2 float4::ra() const noexcept { return float2(x, w); }
    float2 float4::gb() const noexcept { return float2(y, z); }
    float2 float4::ga() const noexcept { return float2(y, w); }
    float2 float4::ba() const noexcept { return float2(z, w); }

    float3 float4::rgb() const noexcept { return float3(x, y, z); }
    float3 float4::rga() const noexcept { return float3(x, y, w); }
    float3 float4::rba() const noexcept { return float3(x, z, w); }
    float3 float4::gba() const noexcept { return float3(y, z, w); }

    float4 float4::grba() const noexcept { return float4(y, x, z, w); }
    float4 float4::brga() const noexcept { return float4(z, x, y, w); }
    float4 float4::bgra() const noexcept { return float4(z, y, x, w); }
    float4 float4::abgr() const noexcept { return float4(w, z, y, x); }

    // ============================================================================
    // Utility Methods Implementation
    // ============================================================================

    bool float4::isValid() const noexcept {
        __m128 zero = _mm_setzero_ps();
        __m128 inf = _mm_set1_ps(std::numeric_limits<float>::infinity());

        __m128 is_nan = _mm_cmpunord_ps(simd_, simd_);
        __m128 is_inf = _mm_cmpeq_ps(_mm_andnot_ps(_mm_set1_ps(-0.0f), simd_), inf);

        return (_mm_movemask_ps(_mm_or_ps(is_nan, is_inf)) == 0);
    }

    bool float4::approximately(const float4& other, float epsilon) const noexcept {
        __m128 diff = _mm_sub_ps(simd_, other.simd_);
        __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff);
        __m128 epsilon_vec = _mm_set1_ps(epsilon);
        __m128 cmp = _mm_cmple_ps(abs_diff, epsilon_vec);
        return (_mm_movemask_ps(cmp) & 0xF) == 0xF;
    }

    bool float4::approximately_zero(float epsilon) const noexcept {
        return length_sq() <= epsilon * epsilon;
    }

    bool float4::is_normalized(float epsilon) const noexcept {
        float len_sq = length_sq();
        return std::isfinite(len_sq) && MathFunctions::approximately(len_sq, 1.0f, epsilon);
    }

    std::string float4::to_string() const {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "(%.3f, %.3f, %.3f, %.3f)", x, y, z, w);
        return std::string(buffer);
    }

    const float* float4::data() const noexcept { return &x; }

    float* float4::data() noexcept { return &x; }

    void float4::set_xyz(const float3& xyz) noexcept {
        __m128 xyz_vec = _mm_set_ps(0.0f, xyz.z, xyz.y, xyz.x);
        simd_ = _mm_blend_ps(simd_, xyz_vec, 0x7); // �������� xyz, ��������� w
    }

    void float4::set_xy(const float2& xy) noexcept {
        __m128 xy_vec = _mm_set_ps(0.0f, 0.0f, xy.y, xy.x);
        simd_ = _mm_blend_ps(simd_, xy_vec, 0x3); // �������� xy, ��������� zw
    }

    void float4::set_zw(const float2& zw) noexcept {
        __m128 zw_vec = _mm_set_ps(zw.y, zw.x, 0.0f, 0.0f);  // w=zw.y, z=zw.x, y=0, x=0

        // ����� 0xC (1100) �������� ���������� w � z
        simd_ = _mm_blend_ps(simd_, zw_vec, 0xC); // �������� z � w, ��������� x � y
    }

    // ============================================================================
    // SSE-specific Methods Implementation
    // ============================================================================

    __m128 float4::get_simd() const noexcept { return simd_; }

    void float4::set_simd(__m128 new_simd) noexcept { simd_ = new_simd; }

    float4 float4::load_unaligned(const float* data) noexcept {
        return float4(_mm_loadu_ps(data));
    }

    float4 float4::load_aligned(const float* data) noexcept {
        return float4(_mm_load_ps(data));
    }

    void float4::store_unaligned(float* data) const noexcept {
        _mm_storeu_ps(data, simd_);
    }

    void float4::store_aligned(float* data) const noexcept {
        _mm_store_ps(data, simd_);
    }

    // ============================================================================
    // Comparison Operators Implementation
    // ============================================================================

    bool float4::operator==(const float4& rhs) const noexcept {
        return approximately(rhs);
    }

    bool float4::operator!=(const float4& rhs) const noexcept {
        return !(*this == rhs);
    }

    // ============================================================================
    // Global Constants Implementation
    // ============================================================================

    const float4 float4_Zero(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 float4_One(1.0f, 1.0f, 1.0f, 1.0f);
    const float4 float4_UnitX(1.0f, 0.0f, 0.0f, 0.0f);
    const float4 float4_UnitY(0.0f, 1.0f, 0.0f, 0.0f);
    const float4 float4_UnitZ(0.0f, 0.0f, 1.0f, 0.0f);
    const float4 float4_UnitW(0.0f, 0.0f, 0.0f, 1.0f);

    // Color constants
    const float4 float4_Red(1.0f, 0.0f, 0.0f, 1.0f);
    const float4 float4_Green(0.0f, 1.0f, 0.0f, 1.0f);
    const float4 float4_Blue(0.0f, 0.0f, 1.0f, 1.0f);
    const float4 float4_White(1.0f, 1.0f, 1.0f, 1.0f);
    const float4 float4_Black(0.0f, 0.0f, 0.0f, 1.0f);
    const float4 float4_Transparent(0.0f, 0.0f, 0.0f, 0.0f);
    const float4 float4_Yellow(1.0f, 1.0f, 0.0f, 1.0f);
    const float4 float4_Cyan(0.0f, 1.0f, 1.0f, 1.0f);
    const float4 float4_Magenta(1.0f, 0.0f, 1.0f, 1.0f);

} // namespace Math
