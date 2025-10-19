/**
 * @file math_half2.cpp
 * @brief Implementation of 2-dimensional half-precision vector class
 * @note Optimized for texture coordinates and memory-constrained applications
 */

#include "math_half2.h"

namespace Math {

    // ============================================================================
    // Constructors Implementation
    // ============================================================================

    half2::half2() noexcept : x(half::from_bits(0)), y(half::from_bits(0)) {}

    half2::half2(half x, half y) noexcept : x(x), y(y) {}

    half2::half2(half scalar) noexcept : x(scalar), y(scalar) {}

    half2::half2(float x, float y) noexcept : x(x), y(y) {}

    half2::half2(float scalar) noexcept : x(scalar), y(scalar) {}

    half2::half2(const float2& vec) noexcept : x(vec.x), y(vec.y) {}

    // ============================================================================
    // Assignment Operators Implementation
    // ============================================================================

    half2& half2::operator=(const float2& vec) noexcept
    {
        x = vec.x;
        y = vec.y;
        return *this;
    }

    half2& half2::operator=(half scalar) noexcept
    {
        x = y = scalar;
        return *this;
    }

    half2& half2::operator=(float scalar) noexcept
    {
        x = y = scalar;
        return *this;
    }

    // ============================================================================
    // Compound Assignment Operators Implementation
    // ============================================================================

    half2& half2::operator+=(const half2& rhs) noexcept
    {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    half2& half2::operator-=(const half2& rhs) noexcept
    {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    half2& half2::operator*=(const half2& rhs) noexcept
    {
        x *= rhs.x;
        y *= rhs.y;
        return *this;
    }

    half2& half2::operator/=(const half2& rhs) noexcept
    {
        x /= rhs.x;
        y /= rhs.y;
        return *this;
    }

    half2& half2::operator*=(half scalar) noexcept
    {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    half2& half2::operator*=(float scalar) noexcept
    {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    half2& half2::operator/=(half scalar) noexcept
    {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    half2& half2::operator/=(float scalar) noexcept
    {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    // ============================================================================
    // Unary Operators Implementation
    // ============================================================================

    half2 half2::operator+() const noexcept
    {
        return *this;
    }

    half2 half2::operator-() const noexcept
    {
        return half2(-x, -y);
    }

    // ============================================================================
    // Access Operators Implementation
    // ============================================================================

    half& half2::operator[](int index) noexcept
    {
        return (&x)[index];
    }

    const half& half2::operator[](int index) const noexcept
    {
        return (&x)[index];
    }

    // ============================================================================
    // Conversion Operators Implementation
    // ============================================================================

    half2::operator float2() const noexcept
    {
        return float2(float(x), float(y));
    }

    // ============================================================================
    // Static Constructors Implementation
    // ============================================================================

    half2 half2::zero() noexcept
    {
        return half2(half::from_bits(0), half::from_bits(0));
    }

    half2 half2::one() noexcept
    {
        return half2(half::from_bits(0x3C00), half::from_bits(0x3C00));
    }

    half2 half2::unit_x() noexcept
    {
        return half2(half::from_bits(0x3C00), half::from_bits(0));
    }

    half2 half2::unit_y() noexcept
    {
        return half2(half::from_bits(0), half::from_bits(0x3C00));
    }

    half2 half2::uv(half u, half v) noexcept
    {
        return half2(u, v);
    }

    // ============================================================================
    // Mathematical Functions Implementation
    // ============================================================================

    half half2::length() const noexcept
    {
        return sqrt(length_sq());
    }

    half half2::length_sq() const noexcept
    {
        return x * x + y * y;
    }

    half2 half2::normalize() const noexcept
    {
        half len = length();
        if (len.is_zero() || !len.is_finite()) {
            return half2::zero();
        }
        return half2(x / len, y / len);
    }

    half half2::dot(const half2& other) const noexcept
    {
        return half2::dot(*this, other);
    }

    half2 half2::perpendicular() const noexcept
    {
        return half2(-y, x);
    }

    half half2::distance(const half2& other) const noexcept
    {
        return (*this - other).length();
    }

    half half2::distance_sq(const half2& other) const noexcept
    {
        return (*this - other).length_sq();
    }

    // ============================================================================
    // HLSL-like Functions Implementation
    // ============================================================================

    half2 half2::abs() const noexcept
    {
        return half2(Math::abs(x), Math::abs(y));
    }

    half2 half2::sign() const noexcept
    {
        return half2(Math::sign(x), Math::sign(y));
    }

    half2 half2::floor() const noexcept
    {
        return half2(Math::floor(x), Math::floor(y));
    }

    half2 half2::ceil() const noexcept
    {
        return half2(Math::ceil(x), Math::ceil(y));
    }

    half2 half2::round() const noexcept
    {
        return half2(Math::round(x), Math::round(y));
    }

    half2 half2::frac() const noexcept
    {
        return half2(Math::frac(x), Math::frac(y));
    }

    half2 half2::saturate() const noexcept
    {
        return half2::saturate(*this);
    }

    half2 half2::step(half edge) const noexcept
    {
        return half2(Math::step(edge, x), Math::step(edge, y));
    }

    // ============================================================================
    // Static Mathematical Functions Implementation
    // ============================================================================

    half half2::dot(const half2& a, const half2& b) noexcept
    {
        return a.x * b.x + a.y * b.y;
    }

    half2 half2::lerp(const half2& a, const half2& b, half t) noexcept
    {
        return half2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
    }

    half2 half2::lerp(const half2& a, const half2& b, float t) noexcept
    {
        return half2(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
    }

    half2 half2::saturate(const half2& vec) noexcept
    {
        return half2(
            Math::saturate(vec.x),
            Math::saturate(vec.y)
        );
    }

    half2 half2::min(const half2& a, const half2& b) noexcept
    {
        return half2(
            (a.x < b.x) ? a.x : b.x,
            (a.y < b.y) ? a.y : b.y
        );
    }

    half2 half2::max(const half2& a, const half2& b) noexcept
    {
        return half2(
            (a.x > b.x) ? a.x : b.x,
            (a.y > b.y) ? a.y : b.y
        );
    }

    // ============================================================================
    // Swizzle Operations Implementation
    // ============================================================================

    half2 half2::yx() const noexcept
    {
        return half2(y, x);
    }

    half2 half2::xx() const noexcept
    {
        return half2(x, x);
    }

    half2 half2::yy() const noexcept
    {
        return half2(y, y);
    }

    // ============================================================================
    // Texture Coordinate Accessors Implementation
    // ============================================================================

    half half2::u() const noexcept
    {
        return x;
    }

    half half2::v() const noexcept
    {
        return y;
    }

    void half2::set_u(half u) noexcept
    {
        x = u;
    }

    void half2::set_v(half v) noexcept
    {
        y = v;
    }

    // ============================================================================
    // Utility Methods Implementation
    // ============================================================================

    bool half2::is_valid() const noexcept
    {
        return x.is_finite() && y.is_finite();
    }

    bool half2::approximately(const half2& other, float epsilon) const noexcept
    {
        return x.approximately(other.x, epsilon) &&
            y.approximately(other.y, epsilon);
    }

    bool half2::approximately_zero(float epsilon) const noexcept
    {
        return x.approximately_zero(epsilon) &&
            y.approximately_zero(epsilon);
    }

    bool half2::is_normalized(float epsilon) const noexcept
    {
        half len_sq = length_sq();
        float adjusted_epsilon = std::max(epsilon, 0.01f);
        return MathFunctions::approximately(float(len_sq), 1.0f, adjusted_epsilon);
    }

    std::string half2::to_string() const
    {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "(%.3f, %.3f)", float(x), float(y));
        return std::string(buffer);
    }

    const half* half2::data() const noexcept
    {
        return &x;
    }

    half* half2::data() noexcept
    {
        return &x;
    }

    // ============================================================================
    // Comparison Operators Implementation
    // ============================================================================

    bool half2::operator==(const half2& rhs) const noexcept
    {
        return approximately(rhs);
    }

    bool half2::operator!=(const half2& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    // ============================================================================
    // Global Constants Implementation
    // ============================================================================

    const half2 half2_Zero(half_Zero);
    const half2 half2_One(half_One);
    const half2 half2_UnitX(half_One, half_Zero);
    const half2 half2_UnitY(half_Zero, half_One);
    const half2 half2_UV_Zero(half_Zero, half_Zero);
    const half2 half2_UV_One(half_One, half_One);
    const half2 half2_UV_Half(half(0.5f), half(0.5f));
    const half2 half2_Right(half_One, half_Zero);
    const half2 half2_Left(-half_One, half_Zero);
    const half2 half2_Up(half_Zero, half_One);
    const half2 half2_Down(half_Zero, -half_One);

} // namespace Math
