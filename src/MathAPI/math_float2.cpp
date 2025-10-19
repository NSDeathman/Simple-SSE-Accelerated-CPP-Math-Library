// Author: NSDeathman, DeepSeek

/**
 * @file math_float2.cpp
 * @brief Implementation of 2-dimensional vector class
 * @note SSE-optimized implementation for performance
 */

#include "math_float2.h"

namespace Math {

    // ============================================================================
    // Constructors Implementation
    // ============================================================================

    float2::float2(const float* data) noexcept : x(data[0]), y(data[1]) {}

    float2::float2(__m128 simd_) noexcept {
        alignas(16) float data[4];
        _mm_store_ps(data, simd_);
        x = data[0];
        y = data[1];
    }

#if defined(MATH_SUPPORT_D3DX)
    float2::float2(const D3DXVECTOR2& vec) noexcept : x(vec.x), y(vec.y) {}

    float2::float2(const D3DXVECTOR4& vec) noexcept : x(vec.x), y(vec.y) {}

    float2::float2(D3DCOLOR color) noexcept {
        x = ((color >> 16) & 0xFF) / 255.0f; // Red channel
        y = ((color >> 8) & 0xFF) / 255.0f;  // Green channel
    }
#endif

    // ============================================================================
    // Assignment Operators Implementation
    // ============================================================================

    float2& float2::operator=(float scalar) noexcept {
        x = y = scalar;
        return *this;
    }

#if defined(MATH_SUPPORT_D3DX)
    float2& float2::operator=(const D3DXVECTOR2& vec) noexcept {
        x = vec.x;
        y = vec.y;
        return *this;
    }

    float2& float2::operator=(D3DCOLOR color) noexcept {
        x = ((color >> 16) & 0xFF) / 255.0f;
        y = ((color >> 8) & 0xFF) / 255.0f;
        return *this;
    }
#endif

    // ============================================================================
    // Compound Assignment Operators Implementation
    // ============================================================================

    float2& float2::operator+=(const float2& rhs) noexcept {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    float2& float2::operator-=(const float2& rhs) noexcept {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    float2& float2::operator*=(const float2& rhs) noexcept {
        x *= rhs.x;
        y *= rhs.y;
        return *this;
    }

    float2& float2::operator/=(const float2& rhs) noexcept {
        x /= rhs.x;
        y /= rhs.y;
        return *this;
    }

    float2& float2::operator*=(float scalar) noexcept {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    float2& float2::operator/=(float scalar) noexcept {
        float inv = 1.0f / scalar;
        x *= inv;
        y *= inv;
        return *this;
    }


    float2 float2::operator+(const float2& rhs) const noexcept {
        __m128 a = _mm_set_ps(0.0f, 0.0f, y, x);  // [0, 0, y, x]
        __m128 b = _mm_set_ps(0.0f, 0.0f, rhs.y, rhs.x);
        __m128 result = _mm_add_ps(a, b);
        return float2(result);
    }

    float2 float2::operator-(const float2& rhs) const noexcept {
        __m128 a = _mm_set_ps(0.0f, 0.0f, y, x);
        __m128 b = _mm_set_ps(0.0f, 0.0f, rhs.y, rhs.x);
        __m128 result = _mm_sub_ps(a, b);
        return float2(result);
    }

    float2 float2::operator+(const float& rhs) const noexcept {
        __m128 a = _mm_set_ps(0.0f, 0.0f, y, x);  // [0, 0, y, x]
        __m128 b = _mm_set_ps(0.0f, 0.0f, rhs, rhs);
        __m128 result = _mm_add_ps(a, b);
        return float2(result);
    }

    float2 float2::operator-(const float& rhs) const noexcept {
        __m128 a = _mm_set_ps(0.0f, 0.0f, y, x);
        __m128 b = _mm_set_ps(0.0f, 0.0f, rhs, rhs);
        __m128 result = _mm_sub_ps(a, b);
        return float2(result);
    }

    // ============================================================================
    // Access Operators Implementation
    // ============================================================================

    float& float2::operator[](int index) noexcept {
        return (&x)[index];
    }

    const float& float2::operator[](int index) const noexcept {
        return (&x)[index];
    }

    // ============================================================================
    // Conversion Operators Implementation
    // ============================================================================

    float2::operator const float* () const noexcept { return &x; }

    float2::operator float* () noexcept { return &x; }

    float2::operator __m128() const noexcept {
        return _mm_set_ps(0.0f, 0.0f, y, x);
    }

#if defined(MATH_SUPPORT_D3DX)
    float2::operator D3DXVECTOR2() const noexcept {
        return D3DXVECTOR2(x, y);
    }
#endif

    // ============================================================================
    // Mathematical Functions Implementation
    // ============================================================================

    float float2::length() const noexcept {
        __m128 vec = _mm_set_ps(0.0f, 0.0f, y, x);

        __m128 squared = _mm_mul_ps(vec, vec);
        __m128 sum = _mm_hadd_ps(squared, squared);

        __m128 sqrt_val = _mm_sqrt_ss(sum);
        return _mm_cvtss_f32(sqrt_val);
    }

    float2 float2::normalize() const noexcept {
        float len = length();
        if (len < Constants::Constants<float>::Epsilon) {
            return float2::zero();
        }

        float inv_len = 1.0f / len;
        __m128 vec = _mm_set_ps(0.0f, 0.0f, y, x);
        __m128 scale = _mm_set1_ps(inv_len);
        __m128 result = _mm_mul_ps(vec, scale);

        return float2(result);
    }

    // ============================================================================
    // HLSL-like Functions Implementation
    // ============================================================================

    float2 float2::abs() const noexcept {
        return float2(std::abs(x), std::abs(y));
    }

    float2 float2::sign() const noexcept {
        return float2(
            (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f),
            (y > 0.0f) ? 1.0f : ((y < 0.0f) ? -1.0f : 0.0f)
        );
    }

    float2 float2::floor() const noexcept {
        return float2(std::floor(x), std::floor(y));
    }

    float2 float2::ceil() const noexcept {
        return float2(std::ceil(x), std::ceil(y));
    }

    float2 float2::round() const noexcept {
        return float2(std::round(x), std::round(y));
    }

    float2 float2::frac() const noexcept {
        return float2(x - std::floor(x), y - std::floor(y));
    }

    float2 float2::saturate() const noexcept {
        return float2(
            std::max(0.0f, std::min(1.0f, x)),
            std::max(0.0f, std::min(1.0f, y))
        );
    }

    float2 float2::step(float edge) const noexcept {
        return float2(
            (x >= edge) ? 1.0f : 0.0f,
            (y >= edge) ? 1.0f : 0.0f
        );
    }

    float2 float2::smoothstep(float edge0, float edge1) const noexcept {
        auto smooth = [](float t, float edge0, float edge1) {
            t = std::max(0.0f, std::min(1.0f, (t - edge0) / (edge1 - edge0)));
            return t * t * (3.0f - 2.0f * t);
        };
        return float2(smooth(x, edge0, edge1), smooth(y, edge0, edge1));
    }

    // ============================================================================
    // Geometric Operations Implementation
    // ============================================================================

    float2 float2::reflect(const float2& normal) const noexcept {
        float dot_val = dot(normal);
        return *this - normal * (2.0f * dot_val);
    }

    float2 float2::refract(const float2& normal, float eta) const noexcept {
        float dot_ni = dot(normal);
        float k = 1.0f - eta * eta * (1.0f - dot_ni * dot_ni);

        if (k < 0.0f)
            return float2::zero(); // total internal reflection

        return *this * eta - normal * (eta * dot_ni + std::sqrt(k));
    }

    float2 float2::rotate(float angle) const noexcept
    {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);

        return float2(
            x * c - y * s,
            x * s + y * c
        );
    }

    float float2::angle() const noexcept {
        return std::atan2(y, x);
    }

    // ============================================================================
    // Static Mathematical Functions Implementation
    // ============================================================================

    float2 float2::slerp(const float2& a, const float2& b, float t) noexcept {
        float dot_val = dot(a, b);
        dot_val = MathFunctions::clamp(dot_val, -1.0f, 1.0f);

        float theta = std::acos(dot_val) * t;
        float2 relative_vec = (b - a * dot_val).normalize();

        return a * std::cos(theta) + relative_vec * std::sin(theta);
    }

    // ============================================================================
    // Utility Methods Implementation
    // ============================================================================

    bool float2::isValid() const noexcept {
        return std::isfinite(x) && std::isfinite(y);
    }

    bool float2::approximately(const float2& other, float epsilon) const noexcept {
        return MathFunctions::approximately(x, other.x, epsilon) &&
            MathFunctions::approximately(y, other.y, epsilon);
    }

    bool float2::approximately_zero(float epsilon) const noexcept {
        return length_sq() <= epsilon * epsilon;
    }

    bool float2::is_normalized(float epsilon) const noexcept
    {
        if (!isValid()) {
            return false;
        }

        float len_sq = length_sq();
        if (!std::isfinite(len_sq)) {
            return false;
        }

        return MathFunctions::approximately(len_sq, 1.0f, epsilon * epsilon);
    }

    std::string float2::to_string() const {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "(%.3f, %.3f)", x, y);
        return std::string(buffer);
    }

    // ============================================================================
    // Comparison Operators Implementation
    // ============================================================================

    bool float2::operator==(const float2& rhs) const noexcept {
        return approximately(rhs);
    }

    bool float2::operator!=(const float2& rhs) const noexcept {
        return !(*this == rhs);
    }

    // ============================================================================
    // Global Constants Implementation
    // ============================================================================

    const float2 float2_Zero(0.0f, 0.0f);
    const float2 float2_One(1.0f, 1.0f);
    const float2 float2_UnitX(1.0f, 0.0f);
    const float2 float2_UnitY(0.0f, 1.0f);
    const float2 float2_Right(1.0f, 0.0f);
    const float2 float2_Left(-1.0f, 0.0f);
    const float2 float2_Up(0.0f, 1.0f);
    const float2 float2_Down(0.0f, -1.0f);

} // namespace Math
