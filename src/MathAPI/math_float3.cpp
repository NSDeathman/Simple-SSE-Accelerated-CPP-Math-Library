// Description: 3-dimensional vector class with comprehensive 
//              mathematical operations and SSE optimization
// Author: NSDeathman, DeepSeek

#include "math_float3.h"

namespace Math
{
    // ============================================================================
    // Constructors Implementation
    // ============================================================================

    float3::float3() noexcept : simd_(_mm_setzero_ps()) {}

    float3::float3(float x, float y, float z) noexcept : simd_(_mm_set_ps(0.0f, z, y, x)) {}

    float3::float3(float scalar) noexcept : simd_(_mm_set_ps(0.0f, scalar, scalar, scalar)) {}

    float3::float3(const float2& vec, float z) noexcept : simd_(_mm_set_ps(0.0f, z, vec.y, vec.x)) {}

    float3::float3(const float* data) noexcept : simd_(_mm_set_ps(0.0f, data[2], data[1], data[0])) {}

    float3::float3(__m128 simd_val) noexcept : simd_(simd_val) {}

#if defined(MATH_SUPPORT_D3DX)
    float3::float3(const D3DXVECTOR3& vec) noexcept : simd_(_mm_set_ps(0.0f, vec.z, vec.y, vec.x)) {}

    float3::float3(const D3DXVECTOR4& vec) noexcept : simd_(_mm_set_ps(0.0f, vec.z, vec.y, vec.x)) {}

    float3::float3(const D3DXVECTOR2& vec, float z) noexcept : simd_(_mm_set_ps(0.0f, z, vec.y, vec.x)) {}

    float3::float3(D3DCOLOR color) noexcept {
        float r = static_cast<float>((color >> 16) & 0xFF) / 255.0f;
        float g = static_cast<float>((color >> 8) & 0xFF) / 255.0f;
        float b = static_cast<float>(color & 0xFF) / 255.0f;
        simd_ = _mm_set_ps(0.0f, b, g, r);
    }
#endif

    // ============================================================================
    // Assignment Operators Implementation
    // ============================================================================

    float3& float3::operator=(float scalar) noexcept {
        simd_ = _mm_set_ps(0.0f, scalar, scalar, scalar);
        return *this;
    }

#if defined(MATH_SUPPORT_D3DX)
    float3& float3::operator=(const D3DXVECTOR3& vec) noexcept {
        simd_ = _mm_set_ps(0.0f, vec.z, vec.y, vec.x);
        return *this;
    }

    float3& float3::operator=(D3DCOLOR color) noexcept {
        float r = static_cast<float>((color >> 16) & 0xFF) / 255.0f;
        float g = static_cast<float>((color >> 8) & 0xFF) / 255.0f;
        float b = static_cast<float>(color & 0xFF) / 255.0f;
        simd_ = _mm_set_ps(0.0f, b, g, r);
        return *this;
    }
#endif

    // ============================================================================
    // Compound Assignment Operators Implementation
    // ============================================================================

    float3& float3::operator+=(const float3& rhs) noexcept {
        simd_ = _mm_add_ps(simd_, rhs.simd_);
        return *this;
    }

    float3& float3::operator-=(const float3& rhs) noexcept {
        simd_ = _mm_sub_ps(simd_, rhs.simd_);
        return *this;
    }

    float3& float3::operator*=(const float3& rhs) noexcept {
        simd_ = _mm_mul_ps(simd_, rhs.simd_);
        return *this;
    }

    float3& float3::operator/=(const float3& rhs) noexcept {
        simd_ = _mm_div_ps(simd_, rhs.simd_);
        return *this;
    }

    float3& float3::operator*=(float scalar) noexcept {
        __m128 scalar_vec = _mm_set1_ps(scalar);
        simd_ = _mm_mul_ps(simd_, scalar_vec);
        return *this;
    }

    float3& float3::operator/=(float scalar) noexcept {
        __m128 inv_scalar = _mm_set1_ps(1.0f / scalar);
        simd_ = _mm_mul_ps(simd_, inv_scalar);
        return *this;
    }

    // ============================================================================
    // Unary Operators Implementation
    // ============================================================================

    float3 float3::operator+() const noexcept {
        return *this;
    }

    float3 float3::operator-() const noexcept {
        __m128 neg = _mm_set1_ps(-1.0f);
        return float3(_mm_mul_ps(simd_, neg));
    }

    // ============================================================================
    // Access Operators Implementation
    // ============================================================================

    float& float3::operator[](int index) noexcept {
        return (&x)[index];
    }

    const float& float3::operator[](int index) const noexcept {
        return (&x)[index];
    }

    // ============================================================================
    // Conversion Operators Implementation
    // ============================================================================

    float3::operator const float* () const noexcept { return &x; }

    float3::operator float* () noexcept { return &x; }

    float3::operator __m128() const noexcept { return simd_; }

#if defined(MATH_SUPPORT_D3DX)
    float3::operator D3DXVECTOR3() const noexcept {
        return D3DXVECTOR3(x, y, z);
    }
#endif

    // ============================================================================
    // Static Constructors Implementation
    // ============================================================================

    float3 float3::zero() noexcept {
        return float3(0.0f, 0.0f, 0.0f);
    }

    float3 float3::one() noexcept {
        return float3(1.0f, 1.0f, 1.0f);
    }

    float3 float3::unit_x() noexcept {
        return float3(1.0f, 0.0f, 0.0f);
    }

    float3 float3::unit_y() noexcept {
        return float3(0.0f, 1.0f, 0.0f);
    }

    float3 float3::unit_z() noexcept {
        return float3(0.0f, 0.0f, 1.0f);
    }

    float3 float3::forward() noexcept {
        return float3(0.0f, 0.0f, 1.0f);
    }

    float3 float3::up() noexcept {
        return float3(0.0f, 1.0f, 0.0f);
    }

    float3 float3::right() noexcept {
        return float3(1.0f, 0.0f, 0.0f);
    }

    // ============================================================================
    // Mathematical Functions Implementation
    // ============================================================================

    float float3::length() const noexcept {
        __m128 squared = _mm_mul_ps(simd_, simd_);
        // Horizontal addition: x+y+z+0
        __m128 shuf = _mm_movehdup_ps(squared);    // y,y,w,w
        __m128 sums = _mm_add_ps(squared, shuf);   // x+y, y+y, z+w, w+w
        shuf = _mm_movehl_ps(shuf, sums);          // z+w, w+w, y,y
        sums = _mm_add_ss(sums, shuf);             // x+y+z+w, ...
        return _mm_cvtss_f32(_mm_sqrt_ss(sums));
    }

    float float3::length_sq() const noexcept {
        __m128 squared = _mm_mul_ps(simd_, simd_);
        // Horizontal addition
        __m128 shuf = _mm_movehdup_ps(squared);
        __m128 sums = _mm_add_ps(squared, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    float3 float3::normalize() const noexcept {
        float len = length();
        if (len > Constants::Constants<float>::Epsilon) {
            __m128 inv_len = _mm_set1_ps(1.0f / len);
            return float3(_mm_mul_ps(simd_, inv_len));
        }
        return float3::zero();
    }

    float float3::dot(const float3& other) const noexcept {
        return float3::dot(*this, other);
    }

    float3 float3::cross(const float3& other) const noexcept {
        return float3::cross(*this, other);
    }

    float float3::distance(const float3& other) const noexcept {
        return (*this - other).length();
    }

    float float3::distance_sq(const float3& other) const noexcept {
        return (*this - other).length_sq();
    }

    // ============================================================================
    // HLSL-like Functions Implementation
    // ============================================================================

    float3 float3::abs() const noexcept {
        __m128 mask = _mm_set1_ps(-0.0f); // -0.0f = 0x80000000
        return float3(_mm_andnot_ps(mask, simd_));
    }

    float3 float3::sign() const noexcept {
        __m128 zero = _mm_setzero_ps();
        __m128 one = _mm_set1_ps(1.0f);
        __m128 neg_one = _mm_set1_ps(-1.0f);

        __m128 gt_zero = _mm_cmpgt_ps(simd_, zero);
        __m128 lt_zero = _mm_cmplt_ps(simd_, zero);

        __m128 result = _mm_and_ps(gt_zero, one);
        result = _mm_or_ps(result, _mm_and_ps(lt_zero, neg_one));

        return float3(result);
    }

    float3 float3::floor() const noexcept {
        alignas(16) float data[4];
        _mm_store_ps(data, simd_);
        return float3(std::floor(data[0]), std::floor(data[1]), std::floor(data[2]));
    }

    float3 float3::ceil() const noexcept {
        alignas(16) float data[4];
        _mm_store_ps(data, simd_);
        return float3(std::ceil(data[0]), std::ceil(data[1]), std::ceil(data[2]));
    }

    float3 float3::round() const noexcept {
        alignas(16) float data[4];
        _mm_store_ps(data, simd_);
        return float3(std::round(data[0]), std::round(data[1]), std::round(data[2]));
    }

    float3 float3::frac() const noexcept {
        alignas(16) float data[4];
        _mm_store_ps(data, simd_);
        return float3(
            data[0] - std::floor(data[0]),
            data[1] - std::floor(data[1]),
            data[2] - std::floor(data[2])
        );
    }

    float3 float3::saturate() const noexcept {
        return float3::saturate(*this);
    }

    float3 float3::step(float edge) const noexcept {
        __m128 edge_vec = _mm_set1_ps(edge);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 zero = _mm_setzero_ps();

        __m128 cmp = _mm_cmpge_ps(simd_, edge_vec);
        return float3(_mm_or_ps(_mm_and_ps(cmp, one), _mm_andnot_ps(cmp, zero)));
    }

    // ============================================================================
    // Geometric Operations Implementation
    // ============================================================================

    float3 float3::reflect(const float3& normal) const noexcept {
        return float3::reflect(*this, normal);
    }

    float3 float3::refract(const float3& normal, float eta) const noexcept {
        return float3::refract(*this, normal, eta);
    }

    float3 float3::project(const float3& onto) const noexcept {
        float onto_length_sq = onto.length_sq();
        if (onto_length_sq < Constants::Constants<float>::Epsilon) {
            return float3::zero();
        }
        float dot_val = dot(onto);
        return onto * (dot_val / onto_length_sq);
    }

    float3 float3::reject(const float3& from) const noexcept {
        return *this - project(from);
    }

    // ============================================================================
    // Static Mathematical Functions Implementation
    // ============================================================================

    float float3::dot(const float3& a, const float3& b) noexcept {
        __m128 a4 = _mm_set_ps(0, a.z, a.y, a.x);
        __m128 b4 = _mm_set_ps(0, b.z, b.y, b.x);
        __m128 mul = _mm_mul_ps(a4, b4);
        __m128 shuf = _mm_movehl_ps(mul, mul);   // [mul.z, mul.w, mul.z, mul.w]
        __m128 sums = _mm_add_ps(mul, shuf);     // x+y, z+w, x+y, z+w
        shuf = _mm_shuffle_ps(sums, sums, 1);    // [z+w, ...]
        sums = _mm_add_ss(sums, shuf);           // x+y+z+w
        return _mm_cvtss_f32(sums);
    }

    float3 float3::cross(const float3& a, const float3& b) noexcept {
        // a.y * b.z - a.z * b.y
        // a.z * b.x - a.x * b.z  
        // a.x * b.y - a.y * b.x

        __m128 a_yzx = _mm_shuffle_ps(a.simd_, a.simd_, _MM_SHUFFLE(3, 0, 2, 1)); // a.y, a.z, a.x, a.w
        __m128 b_yzx = _mm_shuffle_ps(b.simd_, b.simd_, _MM_SHUFFLE(3, 0, 2, 1)); // b.y, b.z, b.x, b.w
        __m128 a_zxy = _mm_shuffle_ps(a.simd_, a.simd_, _MM_SHUFFLE(3, 1, 0, 2)); // a.z, a.x, a.y, a.w
        __m128 b_zxy = _mm_shuffle_ps(b.simd_, b.simd_, _MM_SHUFFLE(3, 1, 0, 2)); // b.z, b.x, b.y, b.w

        __m128 mul1 = _mm_mul_ps(a_yzx, b_zxy);
        __m128 mul2 = _mm_mul_ps(a_zxy, b_yzx);
        __m128 result = _mm_sub_ps(mul1, mul2);

        return float3(result);
    }

    float3 float3::lerp(const float3& a, const float3& b, float t) noexcept {
        __m128 t_vec = _mm_set1_ps(t);
        __m128 one_minus_t = _mm_set1_ps(1.0f - t);

        __m128 part1 = _mm_mul_ps(a.simd_, one_minus_t);
        __m128 part2 = _mm_mul_ps(b.simd_, t_vec);

        return float3(_mm_add_ps(part1, part2));
    }

    float3 float3::slerp(const float3& a, const float3& b, float t) noexcept {
        // Handle edge cases
        if (t <= 0.0f) return a;
        if (t >= 1.0f) return b;

        // Normalize inputs to ensure they're unit vectors
        float3 a_norm = a.normalize();
        float3 b_norm = b.normalize();

        float dot_val = dot(a_norm, b_norm);
        dot_val = MathFunctions::clamp(dot_val, -1.0f, 1.0f);

        // If vectors are very close, use linear interpolation with normalization
        if (dot_val > 0.9995f) {
            return lerp(a_norm, b_norm, t).normalize();
        }

        // Handle case when vectors are opposite (avoid division by zero)
        if (dot_val < -0.9995f) {
            // Choose an arbitrary orthogonal vector
            float3 orthogonal;
            if (std::abs(a_norm.x) > 0.1f) {
                orthogonal = float3(-a_norm.y, a_norm.x, 0.0f).normalize();
            }
            else {
                orthogonal = float3(0.0f, -a_norm.z, a_norm.y).normalize();
            }
            float angle = Constants::PI * t;
            return a_norm * std::cos(angle) + orthogonal * std::sin(angle);
        }

        // Standard slerp implementation
        float theta = std::acos(dot_val);
        float sin_theta = std::sin(theta);

        // Avoid division by zero
        if (sin_theta < 1e-8f) {
            return lerp(a_norm, b_norm, t).normalize();
        }

        float factor_a = std::sin((1.0f - t) * theta) / sin_theta;
        float factor_b = std::sin(t * theta) / sin_theta;

        return (a_norm * factor_a + b_norm * factor_b).normalize();
    }

    float3 float3::min(const float3& a, const float3& b) noexcept {
        return float3(_mm_min_ps(a.simd_, b.simd_));
    }

    float3 float3::max(const float3& a, const float3& b) noexcept {
        return float3(_mm_max_ps(a.simd_, b.simd_));
    }

    float3 float3::saturate(const float3& vec) noexcept {
        __m128 zero = _mm_setzero_ps();
        __m128 one = _mm_set1_ps(1.0f);
        __m128 result = _mm_max_ps(vec.simd_, zero);
        result = _mm_min_ps(result, one);
        return float3(result);
    }

    float3 float3::reflect(const float3& incident, const float3& normal) noexcept {
        float dot_val = dot(incident, normal);
        __m128 dot_vec = _mm_set1_ps(2.0f * dot_val);
        __m128 reflection = _mm_mul_ps(normal.simd_, dot_vec);
        reflection = _mm_sub_ps(incident.simd_, reflection);
        return float3(reflection);
    }

    float3 float3::refract(const float3& incident, const float3& normal, float eta) noexcept {
        float dot_ni = dot(normal, incident);
        float k = 1.0f - eta * eta * (1.0f - dot_ni * dot_ni);

        if (k < 0.0f)
            return float3::zero(); // total internal reflection

        __m128 eta_vec = _mm_set1_ps(eta);
        __m128 incident_eta = _mm_mul_ps(incident.simd_, eta_vec);

        float sqrt_k = std::sqrt(k);
        __m128 dot_eta = _mm_set1_ps(eta * dot_ni + sqrt_k);
        __m128 normal_part = _mm_mul_ps(normal.simd_, dot_eta);

        return float3(_mm_sub_ps(incident_eta, normal_part));
    }

    // ============================================================================
    // Swizzle Operations Implementation
    // ============================================================================

    float2 float3::xy() const noexcept { return float2(x, y); }
    float2 float3::xz() const noexcept { return float2(x, z); }
    float2 float3::yz() const noexcept { return float2(y, z); }
    float2 float3::yx() const noexcept { return float2(y, x); }
    float2 float3::zx() const noexcept { return float2(z, x); }
    float2 float3::zy() const noexcept { return float2(z, y); }

    float3 float3::yxz() const noexcept { return float3(y, x, z); }
    float3 float3::zxy() const noexcept { return float3(z, x, y); }
    float3 float3::zyx() const noexcept { return float3(z, y, x); }
    float3 float3::xzy() const noexcept { return float3(x, z, y); }

    float3 float3::xyx() const noexcept { return float3(x, y, x); }
    float3 float3::xyz() const noexcept { return float3(x, y, z); }
    float3 float3::xzx() const noexcept { return float3(x, z, x); }
    float3 float3::yxy() const noexcept { return float3(y, x, y); }
    float3 float3::yzy() const noexcept { return float3(y, z, y); }
    float3 float3::zxz() const noexcept { return float3(z, x, z); }
    float3 float3::zyz() const noexcept { return float3(z, y, z); }

    // Color swizzles
    float float3::r() const noexcept { return x; }
    float float3::g() const noexcept { return y; }
    float float3::b() const noexcept { return z; }
    float2 float3::rg() const noexcept { return float2(x, y); }
    float2 float3::rb() const noexcept { return float2(x, z); }
    float2 float3::gb() const noexcept { return float2(y, z); }
    float3 float3::rgb() const noexcept { return *this; }
    float3 float3::bgr() const noexcept { return float3(z, y, x); }
    float3 float3::gbr() const noexcept { return float3(y, z, x); }

    // ============================================================================
    // Utility Methods Implementation
    // ============================================================================

    bool float3::isValid() const noexcept {
        return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
    }

    bool float3::approximately(const float3& other, float epsilon) const noexcept {
        __m128 diff = _mm_sub_ps(simd_, other.simd_);
        __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0f), diff); // absolute value
        __m128 epsilon_vec = _mm_set1_ps(epsilon);
        __m128 cmp = _mm_cmple_ps(abs_diff, epsilon_vec);

        // Check that all three components satisfy the condition
        return (_mm_movemask_ps(cmp) & 0x7) == 0x7;
    }

    bool float3::approximately_zero(float epsilon) const noexcept {
        return length_sq() <= epsilon * epsilon;
    }

    bool float3::is_normalized(float epsilon) const noexcept {
        if (!isValid()) {
            return false;
        }

        float len_sq = length_sq();
        if (!std::isfinite(len_sq)) {
            return false;
        }

        return MathFunctions::approximately(len_sq, 1.0f, epsilon * epsilon);
    }

    std::string float3::to_string() const {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "(%.3f, %.3f, %.3f)", x, y, z);
        return std::string(buffer);
    }

    const float* float3::data() const noexcept { return &x; }

    float* float3::data() noexcept { return &x; }

    void float3::set_xy(const float2& xy) noexcept {
        x = xy.x;
        y = xy.y;
    }

    __m128 float3::get_simd() const noexcept { return simd_; }

    void float3::set_simd(__m128 new_simd) noexcept { simd_ = new_simd; }

    // ============================================================================
    // Comparison Operators Implementation
    // ============================================================================

    bool float3::operator==(const float3& rhs) const noexcept {
        return approximately(rhs);
    }

    bool float3::operator!=(const float3& rhs) const noexcept {
        return !(*this == rhs);
    }

    // ============================================================================
    // Global Constants Implementation
    // ============================================================================

    const float3 float3_Zero(0.0f, 0.0f, 0.0f);
    const float3 float3_One(1.0f, 1.0f, 1.0f);
    const float3 float3_UnitX(1.0f, 0.0f, 0.0f);
    const float3 float3_UnitY(0.0f, 1.0f, 0.0f);
    const float3 float3_UnitZ(0.0f, 0.0f, 1.0f);
    const float3 float3_Forward(0.0f, 0.0f, 1.0f);
    const float3 float3_Up(0.0f, 1.0f, 0.0f);
    const float3 float3_Right(1.0f, 0.0f, 0.0f);

} // namespace Math
