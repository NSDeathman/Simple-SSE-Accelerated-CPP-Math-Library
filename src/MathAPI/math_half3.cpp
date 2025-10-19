/**
 * @file math_half3.cpp
 * @brief Implementation of 3-dimensional half-precision vector class
 * @note Optimized for 3D graphics, normals, colors with SSE optimization
 */

#include "math_half3.h"

namespace Math {

    // ============================================================================
    // Constructors Implementation
    // ============================================================================

    half3::half3() noexcept : x(half::from_bits(0)), y(half::from_bits(0)), z(half::from_bits(0)) {}

    half3::half3(half x, half y, half z) noexcept : x(x), y(y), z(z) {}

    half3::half3(half scalar) noexcept : x(scalar), y(scalar), z(scalar) {}

    half3::half3(float x, float y, float z) noexcept : x(x), y(y), z(z) {}

    half3::half3(float scalar) noexcept : x(scalar), y(scalar), z(scalar) {}

    half3::half3(const half2& vec, half z) noexcept : x(vec.x), y(vec.y), z(z) {}

    half3::half3(const float3& vec) noexcept : x(vec.x), y(vec.y), z(vec.z) {}

    half3::half3(const float2& vec, float z) noexcept : x(vec.x), y(vec.y), z(z) {}

    // ============================================================================
    // Assignment Operators Implementation
    // ============================================================================

    half3& half3::operator=(const float3& vec) noexcept
    {
        x = vec.x;
        y = vec.y;
        z = vec.z;
        return *this;
    }

    half3& half3::operator=(half scalar) noexcept
    {
        x = y = z = scalar;
        return *this;
    }

    half3& half3::operator=(float scalar) noexcept
    {
        x = y = z = scalar;
        return *this;
    }

    // ============================================================================
    // Compound Assignment Operators Implementation
    // ============================================================================

    half3& half3::operator+=(const half3& rhs) noexcept
    {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    half3& half3::operator-=(const half3& rhs) noexcept
    {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    half3& half3::operator*=(const half3& rhs) noexcept
    {
        x *= rhs.x;
        y *= rhs.y;
        z *= rhs.z;
        return *this;
    }

    half3& half3::operator/=(const half3& rhs) noexcept
    {
        x /= rhs.x;
        y /= rhs.y;
        z /= rhs.z;
        return *this;
    }

    half3& half3::operator*=(half scalar) noexcept
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    half3& half3::operator*=(float scalar) noexcept
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    half3& half3::operator/=(half scalar) noexcept
    {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    half3& half3::operator/=(float scalar) noexcept
    {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    // ============================================================================
    // Unary Operators Implementation
    // ============================================================================

    half3 half3::operator+() const noexcept
    {
        return *this;
    }

    half3 half3::operator-() const noexcept
    {
        return half3(-x, -y, -z);
    }

    // ============================================================================
    // Access Operators Implementation
    // ============================================================================

    half& half3::operator[](int index) noexcept
    {
        return (&x)[index];
    }

    const half& half3::operator[](int index) const noexcept
    {
        return (&x)[index];
    }

    // ============================================================================
    // Conversion Operators Implementation
    // ============================================================================

    half3::operator float3() const noexcept
    {
        return float3(float(x), float(y), float(z));
    }

    // ============================================================================
    // Static Constructors Implementation
    // ============================================================================

    half3 half3::zero() noexcept
    {
        return half3(half::from_bits(0), half::from_bits(0), half::from_bits(0));
    }

    half3 half3::one() noexcept
    {
        return half3(half::from_bits(0x3C00), half::from_bits(0x3C00), half::from_bits(0x3C00));
    }

    half3 half3::unit_x() noexcept
    {
        return half3(half::from_bits(0x3C00), half::from_bits(0), half::from_bits(0));
    }

    half3 half3::unit_y() noexcept
    {
        return half3(half::from_bits(0), half::from_bits(0x3C00), half::from_bits(0));
    }

    half3 half3::unit_z() noexcept
    {
        return half3(half::from_bits(0), half::from_bits(0), half::from_bits(0x3C00));
    }

    half3 half3::forward() noexcept
    {
        return unit_z();
    }

    half3 half3::up() noexcept
    {
        return unit_y();
    }

    half3 half3::right() noexcept
    {
        return unit_x();
    }

    // ============================================================================
    // Mathematical Functions Implementation
    // ============================================================================

    half half3::length() const noexcept
    {
        float fx = float(x);
        float fy = float(y);
        float fz = float(z);
        return half(std::sqrt(fx * fx + fy * fy + fz * fz));
    }

    half half3::length_sq() const noexcept
    {
        float fx = float(x);
        float fy = float(y);
        float fz = float(z);
        return half(fx * fx + fy * fy + fz * fz);
    }

    half3 half3::normalize() const noexcept
    {
        half len = length();

        if (len.approximately_zero(Constants::Constants<float>::Epsilon * 10.0f))
            return half3::zero();

        half inv_len = half(1.0f) / len;
        return half3(x * inv_len, y * inv_len, z * inv_len);
    }

    half half3::dot(const half3& other) const noexcept
    {
        return half3::dot(*this, other);
    }

    half3 half3::cross(const half3& other) const noexcept
    {
        return half3::cross(*this, other);
    }

    half half3::distance(const half3& other) const noexcept
    {
        half3 diff = *this - other;
        return diff.length();
    }

    half half3::distance_sq(const half3& other) const noexcept
    {
        half3 diff = *this - other;
        return diff.length_sq();
    }

    // ============================================================================
    // HLSL-like Functions Implementation
    // ============================================================================

    half3 half3::abs() const noexcept
    {
        return half3(Math::abs(x), Math::abs(y), Math::abs(z));
    }

    half3 half3::sign() const noexcept
    {
        return half3(Math::sign(x), Math::sign(y), Math::sign(z));
    }

    half3 half3::floor() const noexcept
    {
        return half3(Math::floor(x), Math::floor(y), Math::floor(z));
    }

    half3 half3::ceil() const noexcept
    {
        return half3(Math::ceil(x), Math::ceil(y), Math::ceil(z));
    }

    half3 half3::round() const noexcept
    {
        return half3(Math::round(x), Math::round(y), Math::round(z));
    }

    half3 half3::frac() const noexcept
    {
        return half3(Math::frac(x), Math::frac(y), Math::frac(z));
    }

    half3 half3::saturate() const noexcept
    {
        return half3::saturate(*this);
    }

    half3 half3::step(half edge) const noexcept
    {
        return half3(Math::step(edge, x), Math::step(edge, y), Math::step(edge, z));
    }

    // ============================================================================
    // Geometric Operations Implementation
    // ============================================================================

    half3 half3::reflect(const half3& normal) const noexcept
    {
        return half3::reflect(*this, normal);
    }

    half3 half3::refract(const half3& normal, half eta) const noexcept
    {
        return half3::refract(*this, normal, eta);
    }

    half3 half3::project(const half3& onto) const noexcept
    {
        half onto_length_sq = onto.length_sq();

        if (onto_length_sq.approximately_zero(Constants::Constants<float>::Epsilon * 10.0f))
            return half3::zero();

        half dot_val = dot(onto);

        if (dot_val.is_finite() && onto_length_sq.is_finite()) {
            return onto * (dot_val / onto_length_sq);
        }
        return half3::zero();
    }

    half3 half3::reject(const half3& from) const noexcept
    {
        half3 projected = project(from);
        return *this - projected;
    }

    // ============================================================================
    // Static Mathematical Functions Implementation
    // ============================================================================

    half half3::dot(const half3& a, const half3& b) noexcept
    {
        float result = float(a.x) * float(b.x) +
            float(a.y) * float(b.y) +
            float(a.z) * float(b.z);
        return half(result);
    }

    half3 half3::cross(const half3& a, const half3& b) noexcept
    {
        float x = float(a.y) * float(b.z) - float(a.z) * float(b.y);
        float y = float(a.z) * float(b.x) - float(a.x) * float(b.z);
        float z = float(a.x) * float(b.y) - float(a.y) * float(b.x);

        return half3(half(x), half(y), half(z));
    }

    half3 half3::lerp(const half3& a, const half3& b, half t) noexcept
    {
        __m128 a_vec = _mm_set_ps(0.0f, float(a.z), float(a.y), float(a.x));
        __m128 b_vec = _mm_set_ps(0.0f, float(b.z), float(b.y), float(b.x));
        __m128 t_vec = _mm_set1_ps(float(t));
        __m128 one_minus_t = _mm_set1_ps(1.0f - float(t));

        __m128 part1 = _mm_mul_ps(a_vec, one_minus_t);
        __m128 part2 = _mm_mul_ps(b_vec, t_vec);
        __m128 result = _mm_add_ps(part1, part2);

        alignas(16) float temp[4];
        _mm_store_ps(temp, result);
        return half3(half(temp[0]), half(temp[1]), half(temp[2]));
    }

    half3 half3::lerp(const half3& a, const half3& b, float t) noexcept
    {
        return lerp(a, b, half(t));
    }

    half3 half3::saturate(const half3& vec) noexcept
    {
        return half3(
            Math::saturate(vec.x),
            Math::saturate(vec.y),
            Math::saturate(vec.z)
        );
    }

    half3 half3::min(const half3& a, const half3& b) noexcept
    {
        return half3(
            (a.x < b.x) ? a.x : b.x,
            (a.y < b.y) ? a.y : b.y,
            (a.z < b.z) ? a.z : b.z
        );
    }

    half3 half3::max(const half3& a, const half3& b) noexcept
    {
        return half3(
            (a.x > b.x) ? a.x : b.x,
            (a.y > b.y) ? a.y : b.y,
            (a.z > b.z) ? a.z : b.z
        );
    }

    half3 half3::reflect(const half3& incident, const half3& normal) noexcept
    {
        half dot_val = dot(incident, normal);
        return incident - half(2.0f) * dot_val * normal;
    }

    half3 half3::refract(const half3& incident, const half3& normal, half eta) noexcept
    {
        half dot_ni = dot(normal, incident);
        half k = half(1.0f) - eta * eta * (half(1.0f) - dot_ni * dot_ni);

        if (k < half(0.0f))
            return half3::zero(); // total internal reflection

        return incident * eta - normal * (eta * dot_ni + sqrt(k));
    }

    // ============================================================================
    // Color Operations Implementation
    // ============================================================================

    half half3::luminance() const noexcept
    {
        return half(0.2126f) * x + half(0.7152f) * y + half(0.0722f) * z;
    }

    half3 half3::rgb_to_grayscale() const noexcept
    {
        half luma = luminance();
        return half3(luma, luma, luma);
    }

    half3 half3::gamma_correct(half gamma) const noexcept
    {
        return half3(Math::pow(x, gamma), Math::pow(y, gamma), Math::pow(z, gamma));
    }

    half3 half3::srgb_to_linear() const noexcept
    {
        auto srgb_to_linear_channel = [](half channel) -> half {
            float c = float(channel);
            return (c <= 0.04045f) ? half(c / 12.92f) : half(std::pow((c + 0.055f) / 1.055f, 2.4f));
        };

        return half3(srgb_to_linear_channel(x), srgb_to_linear_channel(y), srgb_to_linear_channel(z));
    }

    half3 half3::linear_to_srgb() const noexcept
    {
        auto linear_to_srgb_channel = [](half channel) -> half {
            float c = float(channel);
            return (c <= 0.0031308f) ? half(c * 12.92f) : half(1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
        };

        return half3(linear_to_srgb_channel(x), linear_to_srgb_channel(y), linear_to_srgb_channel(z));
    }

    // ============================================================================
    // Swizzle Operations Implementation
    // ============================================================================

    half2 half3::xy() const noexcept { return half2(x, y); }
    half2 half3::xz() const noexcept { return half2(x, z); }
    half2 half3::yz() const noexcept { return half2(y, z); }
    half2 half3::yx() const noexcept { return half2(y, x); }
    half2 half3::zx() const noexcept { return half2(z, x); }
    half2 half3::zy() const noexcept { return half2(z, y); }

    half3 half3::yxz() const noexcept { return half3(y, x, z); }
    half3 half3::zxy() const noexcept { return half3(z, x, y); }
    half3 half3::zyx() const noexcept { return half3(z, y, x); }
    half3 half3::xzy() const noexcept { return half3(x, z, y); }

    half half3::r() const noexcept { return x; }
    half half3::g() const noexcept { return y; }
    half half3::b() const noexcept { return z; }
    half2 half3::rg() const noexcept { return half2(x, y); }
    half2 half3::rb() const noexcept { return half2(x, z); }
    half2 half3::gb() const noexcept { return half2(y, z); }
    half3 half3::rgb() const noexcept { return *this; }
    half3 half3::bgr() const noexcept { return half3(z, y, x); }
    half3 half3::gbr() const noexcept { return half3(y, z, x); }

    // ============================================================================
    // Utility Methods Implementation
    // ============================================================================

    bool half3::is_valid() const noexcept
    {
        return x.is_finite() && y.is_finite() && z.is_finite();
    }

    bool half3::approximately(const half3& other, float epsilon) const noexcept
    {
        return x.approximately(other.x, epsilon) &&
            y.approximately(other.y, epsilon) &&
            z.approximately(other.z, epsilon);
    }

    bool half3::approximately_zero(float epsilon) const noexcept
    {
        return std::abs(float(x)) <= epsilon &&
            std::abs(float(y)) <= epsilon &&
            std::abs(float(z)) <= epsilon;
    }

    bool half3::is_normalized(float epsilon) const noexcept
    {
        half len_sq = length_sq();
        float len_sq_f = float(len_sq);
        return std::abs(len_sq_f - 1.0f) <= epsilon;
    }

    std::string half3::to_string() const
    {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "(%.3f, %.3f, %.3f)", float(x), float(y), float(z));
        return std::string(buffer);
    }

    const half* half3::data() const noexcept
    {
        return &x;
    }

    half* half3::data() noexcept
    {
        return &x;
    }

    void half3::set_xy(const half2& xy) noexcept
    {
        x = xy.x;
        y = xy.y;
    }

    // ============================================================================
    // Comparison Operators Implementation
    // ============================================================================

    bool half3::operator==(const half3& rhs) const noexcept
    {
        return approximately(rhs);
    }

    bool half3::operator!=(const half3& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    // ============================================================================
    // Global Constants Implementation
    // ============================================================================

    const half3 half3_Zero(half_Zero);
    const half3 half3_One(half_One);
    const half3 half3_UnitX(half_One, half_Zero, half_Zero);
    const half3 half3_UnitY(half_Zero, half_One, half_Zero);
    const half3 half3_UnitZ(half_Zero, half_Zero, half_One);
    const half3 half3_Forward(half_Zero, half_Zero, half_One);
    const half3 half3_Up(half_Zero, half_One, half_Zero);
    const half3 half3_Right(half_One, half_Zero, half_Zero);
    const half3 half3_Red(half_One, half_Zero, half_Zero);
    const half3 half3_Green(half_Zero, half_One, half_Zero);
    const half3 half3_Blue(half_Zero, half_Zero, half_One);
    const half3 half3_White(half_One);
    const half3 half3_Black(half_Zero);
    const half3 half3_Yellow(half_One, half_One, half_Zero);
    const half3 half3_Cyan(half_Zero, half_One, half_One);
    const half3 half3_Magenta(half_One, half_Zero, half_One);

} // namespace Math
