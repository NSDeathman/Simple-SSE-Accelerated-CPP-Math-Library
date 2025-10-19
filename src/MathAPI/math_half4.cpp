/**
 * @file math_half4.cpp
 * @brief Implementation of 4-dimensional half-precision vector class
 * @note Optimized for 4D graphics, homogeneous coordinates, RGBA colors with SSE optimization
 */

#include "math_half4.h"

namespace Math {

    // ============================================================================
    // Constructors Implementation
    // ============================================================================

    half4::half4() noexcept : x(half::from_bits(0)), y(half::from_bits(0)), z(half::from_bits(0)), w(half::from_bits(0)) {}

    half4::half4(half x, half y, half z, half w) noexcept : x(x), y(y), z(z), w(w) {}

    half4::half4(half scalar) noexcept : x(scalar), y(scalar), z(scalar), w(scalar) {}

    half4::half4(float x, float y, float z, float w) noexcept : x(x), y(y), z(z), w(w) {}

    half4::half4(float scalar) noexcept : x(scalar), y(scalar), z(scalar), w(scalar) {}

    half4::half4(const half2& vec, half z, half w) noexcept
        : x(vec.x), y(vec.y), z(z), w(w) {}

    half4::half4(const half3& vec, half w) noexcept
        : x(vec.x), y(vec.y), z(vec.z), w(w) {}

    half4::half4(const float4& vec) noexcept : x(vec.x), y(vec.y), z(vec.z), w(vec.w) {}

    half4::half4(const float2& vec, float z, float w) noexcept
        : x(vec.x), y(vec.y), z(z), w(w) {}

    half4::half4(const float3& vec, float w) noexcept
        : x(vec.x), y(vec.y), z(vec.z), w(w) {}

    // ============================================================================
    // Assignment Operators Implementation
    // ============================================================================

    half4& half4::operator=(const float4& vec) noexcept
    {
        x = vec.x;
        y = vec.y;
        z = vec.z;
        w = vec.w;
        return *this;
    }

    half4& half4::operator=(const half3& xyz) noexcept
    {
        x = xyz.x;
        y = xyz.y;
        z = xyz.z;
        // w component remains unchanged
        return *this;
    }

    half4& half4::operator=(half scalar) noexcept
    {
        x = y = z = w = scalar;
        return *this;
    }

    half4& half4::operator=(float scalar) noexcept
    {
        x = y = z = w = scalar;
        return *this;
    }

    // ============================================================================
    // Compound Assignment Operators Implementation
    // ============================================================================

    half4& half4::operator+=(const half4& rhs) noexcept
    {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        w += rhs.w;
        return *this;
    }

    half4& half4::operator-=(const half4& rhs) noexcept
    {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        w -= rhs.w;
        return *this;
    }

    half4& half4::operator*=(const half4& rhs) noexcept
    {
        x *= rhs.x;
        y *= rhs.y;
        z *= rhs.z;
        w *= rhs.w;
        return *this;
    }

    half4& half4::operator/=(const half4& rhs) noexcept
    {
        x /= rhs.x;
        y /= rhs.y;
        z /= rhs.z;
        w /= rhs.w;
        return *this;
    }

    half4& half4::operator*=(half scalar) noexcept
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        return *this;
    }

    half4& half4::operator*=(float scalar) noexcept
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        return *this;
    }

    half4& half4::operator/=(half scalar) noexcept
    {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        w /= scalar;
        return *this;
    }

    half4& half4::operator/=(float scalar) noexcept
    {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        w /= scalar;
        return *this;
    }

    // ============================================================================
    // Unary Operators Implementation
    // ============================================================================

    half4 half4::operator+() const noexcept
    {
        return *this;
    }

    half4 half4::operator-() const noexcept
    {
        return half4(-x, -y, -z, -w);
    }

    // ============================================================================
    // Access Operators Implementation
    // ============================================================================

    half& half4::operator[](int index) noexcept
    {
        return (&x)[index];
    }

    const half& half4::operator[](int index) const noexcept
    {
        return (&x)[index];
    }

    // ============================================================================
    // Conversion Operators Implementation
    // ============================================================================

    half4::operator float4() const noexcept
    {
        return float4(float(x), float(y), float(z), float(w));
    }

    // ============================================================================
    // Static Constructors Implementation
    // ============================================================================

    half4 half4::zero() noexcept
    {
        return half4(half::from_bits(0), half::from_bits(0), half::from_bits(0), half::from_bits(0));
    }

    half4 half4::one() noexcept
    {
        return half4(half::from_bits(0x3C00), half::from_bits(0x3C00), half::from_bits(0x3C00), half::from_bits(0x3C00));
    }

    half4 half4::unit_x() noexcept
    {
        return half4(half::from_bits(0x3C00), half::from_bits(0), half::from_bits(0), half::from_bits(0));
    }

    half4 half4::unit_y() noexcept
    {
        return half4(half::from_bits(0), half::from_bits(0x3C00), half::from_bits(0), half::from_bits(0));
    }

    half4 half4::unit_z() noexcept
    {
        return half4(half::from_bits(0), half::from_bits(0), half::from_bits(0x3C00), half::from_bits(0));
    }

    half4 half4::unit_w() noexcept
    {
        return half4(half::from_bits(0), half::from_bits(0), half::from_bits(0), half::from_bits(0x3C00));
    }

    half4 half4::from_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) noexcept
    {
        return half4(
            half(static_cast<float>(r) / 255.0f),
            half(static_cast<float>(g) / 255.0f),
            half(static_cast<float>(b) / 255.0f),
            half(static_cast<float>(a) / 255.0f)
        );
    }

    // ============================================================================
    // Mathematical Functions Implementation
    // ============================================================================

    half half4::length() const noexcept
    {
        float len_sq = float(x) * float(x) + float(y) * float(y) +
            float(z) * float(z) + float(w) * float(w);
        return half(std::sqrt(len_sq));
    }

    half half4::length_sq() const noexcept
    {
        return half(float(x) * float(x) + float(y) * float(y) +
            float(z) * float(z) + float(w) * float(w));
    }

    half4 half4::normalize() const noexcept
    {
        half len = length();
        if (len > half(1e-3f)) {
            return half4(x / len, y / len, z / len, w / len);
        }
        return half4::zero();
    }

    half half4::dot(const half4& other) const noexcept
    {
        return half4::dot(*this, other);
    }

    half half4::dot3(const half4& other) const noexcept
    {
        return half4::dot3(*this, other);
    }

    half4 half4::cross(const half4& other) const noexcept
    {
        return half4(
            half(float(y) * float(other.z) - float(z) * float(other.y)),
            half(float(z) * float(other.x) - float(x) * float(other.z)),
            half(float(x) * float(other.y) - float(y) * float(other.x)),
            half(0.0f)
        );
    }

    half half4::distance(const half4& other) const noexcept
    {
        return (*this - other).length();
    }

    half half4::distance_sq(const half4& other) const noexcept
    {
        return (*this - other).length_sq();
    }

    // ============================================================================
    // HLSL-like Functions Implementation
    // ============================================================================

    half4 half4::abs() const noexcept
    {
        return half4(Math::abs(x), Math::abs(y), Math::abs(z), Math::abs(w));
    }

    half4 half4::sign() const noexcept
    {
        return half4(Math::sign(x), Math::sign(y), Math::sign(z), Math::sign(w));
    }

    half4 half4::floor() const noexcept
    {
        return half4(Math::floor(x), Math::floor(y), Math::floor(z), Math::floor(w));
    }

    half4 half4::ceil() const noexcept
    {
        return half4(Math::ceil(x), Math::ceil(y), Math::ceil(z), Math::ceil(w));
    }

    half4 half4::round() const noexcept
    {
        return half4(Math::round(x), Math::round(y), Math::round(z), Math::round(w));
    }

    half4 half4::frac() const noexcept
    {
        return half4(Math::frac(x), Math::frac(y), Math::frac(z), Math::frac(w));
    }

    half4 half4::saturate() const noexcept
    {
        return half4::saturate(*this);
    }

    half4 half4::step(half edge) const noexcept
    {
        return half4(Math::step(edge, x), Math::step(edge, y), Math::step(edge, z), Math::step(edge, w));
    }

    // ============================================================================
    // Color Operations Implementation
    // ============================================================================

    half half4::luminance() const noexcept
    {
        // SSE optimized luminance calculation
        __m128 vec = _mm_set_ps(0.0f, float(z), float(y), float(x));
        __m128 weights = _mm_set_ps(0.0f, 0.0722f, 0.7152f, 0.2126f);
        __m128 mul = _mm_mul_ps(vec, weights);
        __m128 sum = _mm_hadd_ps(mul, mul);
        sum = _mm_hadd_ps(sum, sum);
        return half(_mm_cvtss_f32(sum));
    }

    half half4::brightness() const noexcept
    {
        // SSE optimized brightness calculation
        __m128 vec = _mm_set_ps(0.0f, float(z), float(y), float(x));
        __m128 sum = _mm_hadd_ps(vec, vec);
        sum = _mm_hadd_ps(sum, sum);
        return half(_mm_cvtss_f32(_mm_mul_ss(sum, _mm_set1_ps(1.0f / 3.0f))));
    }

    half4 half4::premultiply_alpha() const noexcept
    {
        return half4(x * w, y * w, z * w, w);
    }

    half4 half4::unpremultiply_alpha() const noexcept
    {
        // Специальная обработка для alpha = 0
        if (w.is_zero()) {
            // Для alpha = 0, возвращаем исходные значения RGB
            // Это стандартное поведение в графике - при alpha=0 RGB значения сохраняются
            return *this;
        }

        // Для очень маленьких alpha значений, используем безопасное деление
        if (w < half(1e-6f)) {
            // Используем минимальное безопасное значение alpha
            float safe_alpha = std::max(float(w), 1e-6f);
            float inv_alpha = 1.0f / safe_alpha;

            // Ограничиваем результат чтобы избежать переполнения
            return half4(
                half(std::clamp(float(x) * inv_alpha, 0.0f, 1000.0f)),
                half(std::clamp(float(y) * inv_alpha, 0.0f, 1000.0f)),
                half(std::clamp(float(z) * inv_alpha, 0.0f, 1000.0f)),
                w
            );
        }

        // Нормальный случай: alpha достаточно большой
        half inv_alpha = half(1.0f) / w;
        return half4(x * inv_alpha, y * inv_alpha, z * inv_alpha, w);
    }

    half4 half4::grayscale() const noexcept
    {
        half lum = luminance();
        return half4(lum, lum, lum, w);
    }

    half4 half4::srgb_to_linear() const noexcept
    {
        auto srgb_to_linear_channel = [](half channel) -> half {
            float c = float(channel);
            return (c <= 0.04045f) ? half(c / 12.92f) : half(std::pow((c + 0.055f) / 1.055f, 2.4f));
        };

        return half4(srgb_to_linear_channel(x), srgb_to_linear_channel(y),
            srgb_to_linear_channel(z), w); // alpha remains linear
    }

    half4 half4::linear_to_srgb() const noexcept
    {
        auto linear_to_srgb_channel = [](half channel) -> half {
            float c = float(channel);
            return (c <= 0.0031308f) ? half(c * 12.92f) : half(1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
        };

        return half4(linear_to_srgb_channel(x), linear_to_srgb_channel(y),
            linear_to_srgb_channel(z), w); // alpha remains linear
    }

    // ============================================================================
    // Geometric Operations Implementation
    // ============================================================================

    half3 half4::project() const noexcept
    {
        if (std::abs(float(w)) > 1e-6f) {
            half inv_w = half(1.0f) / w;
            return half3(x * inv_w, y * inv_w, z * inv_w);
        }
        return half3::zero();
    }

    half4 half4::to_homogeneous() const noexcept
    {
        return half4(x, y, z, half_One);
    }

    // ============================================================================
    // Static Mathematical Functions Implementation
    // ============================================================================

    half half4::dot(const half4& a, const half4& b) noexcept
    {
        // SSE optimized dot product
        __m128 a_vec = _mm_set_ps(float(a.w), float(a.z), float(a.y), float(a.x));
        __m128 b_vec = _mm_set_ps(float(b.w), float(b.z), float(b.y), float(b.x));
        __m128 mul = _mm_mul_ps(a_vec, b_vec);
        __m128 sum = _mm_hadd_ps(mul, mul);
        sum = _mm_hadd_ps(sum, sum);
        return half(_mm_cvtss_f32(sum));
    }

    half half4::dot3(const half4& a, const half4& b) noexcept
    {
        // SSE optimized 3D dot product
        __m128 a_vec = _mm_set_ps(0.0f, float(a.z), float(a.y), float(a.x));
        __m128 b_vec = _mm_set_ps(0.0f, float(b.z), float(b.y), float(b.x));
        __m128 mul = _mm_mul_ps(a_vec, b_vec);
        __m128 sum = _mm_hadd_ps(mul, mul);
        sum = _mm_hadd_ps(sum, sum);
        return half(_mm_cvtss_f32(sum));
    }

    half4 half4::cross(const half4& a, const half4& b) noexcept
    {
        // SSE optimized cross product
        __m128 a_vec = _mm_set_ps(0.0f, float(a.z), float(a.y), float(a.x));
        __m128 b_vec = _mm_set_ps(0.0f, float(b.z), float(b.y), float(b.x));

        // a.y * b.z - a.z * b.y
        // a.z * b.x - a.x * b.z
        // a.x * b.y - a.y * b.x

        __m128 a_yzx = _mm_shuffle_ps(a_vec, a_vec, _MM_SHUFFLE(3, 0, 2, 1)); // a.y, a.z, a.x, a.w
        __m128 b_yzx = _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(3, 0, 2, 1)); // b.y, b.z, b.x, b.w
        __m128 a_zxy = _mm_shuffle_ps(a_vec, a_vec, _MM_SHUFFLE(3, 1, 0, 2)); // a.z, a.x, a.y, a.w
        __m128 b_zxy = _mm_shuffle_ps(b_vec, b_vec, _MM_SHUFFLE(3, 1, 0, 2)); // b.z, b.x, b.y, b.w

        __m128 mul1 = _mm_mul_ps(a_yzx, b_zxy);
        __m128 mul2 = _mm_mul_ps(a_zxy, b_yzx);
        __m128 result = _mm_sub_ps(mul1, mul2);

        // Extract result components and set w = 0
        alignas(16) float temp[4];
        _mm_store_ps(temp, result);
        return half4(half(temp[0]), half(temp[1]), half(temp[2]), half::from_bits(0));
    }

    half4 half4::lerp(const half4& a, const half4& b, half t) noexcept
    {
        // SSE optimized lerp
        __m128 a_vec = _mm_set_ps(float(a.w), float(a.z), float(a.y), float(a.x));
        __m128 b_vec = _mm_set_ps(float(b.w), float(b.z), float(b.y), float(b.x));
        __m128 t_vec = _mm_set1_ps(float(t));
        __m128 one_minus_t = _mm_set1_ps(1.0f - float(t));

        __m128 part1 = _mm_mul_ps(a_vec, one_minus_t);
        __m128 part2 = _mm_mul_ps(b_vec, t_vec);
        __m128 result = _mm_add_ps(part1, part2);

        alignas(16) float temp[4];
        _mm_store_ps(temp, result);
        return half4(half(temp[0]), half(temp[1]), half(temp[2]), half(temp[3]));
    }

    half4 half4::lerp(const half4& a, const half4& b, float t) noexcept
    {
        return lerp(a, b, half(t));
    }

    half4 half4::min(const half4& a, const half4& b) noexcept
    {
        return half4(
            (a.x < b.x) ? a.x : b.x,
            (a.y < b.y) ? a.y : b.y,
            (a.z < b.z) ? a.z : b.z,
            (a.w < b.w) ? a.w : b.w
        );
    }

    half4 half4::max(const half4& a, const half4& b) noexcept
    {
        return half4(
            (a.x > b.x) ? a.x : b.x,
            (a.y > b.y) ? a.y : b.y,
            (a.z > b.z) ? a.z : b.z,
            (a.w > b.w) ? a.w : b.w
        );
    }

    half4 half4::saturate(const half4& vec) noexcept
    {
        return half4(
            Math::saturate(vec.x),
            Math::saturate(vec.y),
            Math::saturate(vec.z),
            Math::saturate(vec.w)
        );
    }

    // ============================================================================
    // Swizzle Operations Implementation
    // ============================================================================

    half2 half4::xy() const noexcept { return half2(x, y); }
    half2 half4::xz() const noexcept { return half2(x, z); }
    half2 half4::xw() const noexcept { return half2(x, w); }
    half2 half4::yz() const noexcept { return half2(y, z); }
    half2 half4::yw() const noexcept { return half2(y, w); }
    half2 half4::zw() const noexcept { return half2(z, w); }

    half3 half4::xyz() const noexcept { return half3(x, y, z); }
    half3 half4::xyw() const noexcept { return half3(x, y, w); }
    half3 half4::xzw() const noexcept { return half3(x, z, w); }
    half3 half4::yzw() const noexcept { return half3(y, z, w); }

    half4 half4::yxzw() const noexcept { return half4(y, x, z, w); }
    half4 half4::zxyw() const noexcept { return half4(z, x, y, w); }
    half4 half4::zyxw() const noexcept { return half4(z, y, x, w); }
    half4 half4::wzyx() const noexcept { return half4(w, z, y, x); }

    half half4::r() const noexcept { return x; }
    half half4::g() const noexcept { return y; }
    half half4::b() const noexcept { return z; }
    half half4::a() const noexcept { return w; }
    half2 half4::rg() const noexcept { return half2(x, y); }
    half2 half4::rb() const noexcept { return half2(x, z); }
    half2 half4::ra() const noexcept { return half2(x, w); }
    half2 half4::gb() const noexcept { return half2(y, z); }
    half2 half4::ga() const noexcept { return half2(y, w); }
    half2 half4::ba() const noexcept { return half2(z, w); }

    half3 half4::rgb() const noexcept { return half3(x, y, z); }
    half3 half4::rga() const noexcept { return half3(x, y, w); }
    half3 half4::rba() const noexcept { return half3(x, z, w); }
    half3 half4::gba() const noexcept { return half3(y, z, w); }

    half4 half4::grba() const noexcept { return half4(y, x, z, w); }
    half4 half4::brga() const noexcept { return half4(z, x, y, w); }
    half4 half4::bgra() const noexcept { return half4(z, y, x, w); }
    half4 half4::abgr() const noexcept { return half4(w, z, y, x); }

    // ============================================================================
    // Utility Methods Implementation
    // ============================================================================

    bool half4::is_valid() const noexcept
    {
        return !is_nan() && !is_inf();
    }

    bool half4::approximately(const half4& other, float epsilon) const noexcept
    {
        return x.approximately(other.x, epsilon) &&
            y.approximately(other.y, epsilon) &&
            z.approximately(other.z, epsilon) &&
            w.approximately(other.w, epsilon);
    }

    bool half4::approximately_zero(float epsilon) const noexcept
    {
        return std::abs(float(x)) <= epsilon &&
            std::abs(float(y)) <= epsilon &&
            std::abs(float(z)) <= epsilon &&
            std::abs(float(w)) <= epsilon;
    }

    bool half4::is_normalized(float epsilon) const noexcept
    {
        half len_sq = length_sq();
        return std::abs(float(len_sq) - 1.0f) <= epsilon;
    }

    std::string half4::to_string() const
    {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "(%.3f, %.3f, %.3f, %.3f)", float(x), float(y), float(z), float(w));
        return std::string(buffer);
    }

    const half* half4::data() const noexcept
    {
        return &x;
    }

    half* half4::data() noexcept
    {
        return &x;
    }

    void half4::set_xyz(const half3& xyz) noexcept
    {
        x = xyz.x;
        y = xyz.y;
        z = xyz.z;
    }

    void half4::set_xy(const half2& xy) noexcept
    {
        x = xy.x;
        y = xy.y;
    }

    void half4::set_zw(const half2& zw) noexcept
    {
        z = zw.x;
        w = zw.y;
    }

    // ============================================================================
    // Comparison Operators Implementation
    // ============================================================================

    bool half4::operator==(const half4& rhs) const noexcept
    {
        return approximately(rhs);
    }

    bool half4::operator!=(const half4& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    // ============================================================================
    // Global Constants Implementation
    // ============================================================================

    const half4 half4_Zero(half_Zero);
    const half4 half4_One(half_One);
    const half4 half4_UnitX(half_One, half_Zero, half_Zero, half_Zero);
    const half4 half4_UnitY(half_Zero, half_One, half_Zero, half_Zero);
    const half4 half4_UnitZ(half_Zero, half_Zero, half_One, half_Zero);
    const half4 half4_UnitW(half_Zero, half_Zero, half_Zero, half_One);
    const half4 half4_Red(half_One, half_Zero, half_Zero, half_One);
    const half4 half4_Green(half_Zero, half_One, half_Zero, half_One);
    const half4 half4_Blue(half_Zero, half_Zero, half_One, half_One);
    const half4 half4_White(half_One, half_One, half_One, half_One);
    const half4 half4_Black(half_Zero, half_Zero, half_Zero, half_One);
    const half4 half4_Transparent(half_Zero, half_Zero, half_Zero, half_Zero);
    const half4 half4_Yellow(half_One, half_One, half_Zero, half_One);
    const half4 half4_Cyan(half_Zero, half_One, half_One, half_One);
    const half4 half4_Magenta(half_One, half_Zero, half_One, half_One);

} // namespace Math
