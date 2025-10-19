// Author: NSDeathman, DeepSeek
#pragma once

/**
 * @file float2.h
 * @brief 2-dimensional vector class with comprehensive mathematical operations
 * @note Optimized for 2D graphics, UI systems, and texture coordinates
 * @note Includes SSE optimization and HLSL-like functions
 */

#include <string>       // std::string
#include <cstdio>       // snprintf
#include <cmath>        // std::sqrt, std::isfinite
#include <algorithm>    // std::min, std::max
#include <xmmintrin.h>
#include <pmmintrin.h> // SSE3

#include "math_config.h"
#include "math_constants.h"
#include "math_functions.h"

 // Platform-specific support
#if defined(MATH_SUPPORT_D3DX)
#include <d3dx9.h>  // D3DXVECTOR2, D3DXVECTOR4, D3DCOLOR
#endif

namespace Math 
{
    /**
     * @class float2
     * @brief 2-dimensional vector with comprehensive mathematical operations
     *
     * Represents a 2D vector (x, y) with optimized operations for 2D graphics,
     * user interfaces, texture coordinates, and 2D physics simulations.
     *
     * @note Perfect for 2D game development, UI systems, and texture operations
     * @note All operations are optimized and constexpr where possible
     * @note Includes SSE optimization for performance-critical operations
     */
    class MATH_API float2 {
    public:
        // ============================================================================
        // Data Members (Public for Direct Access)
        // ============================================================================

        float x; ///< X component of the vector
        float y; ///< Y component of the vector

        // ============================================================================
        // Constructors
        // ============================================================================

        /**
         * @brief Default constructor (initializes to zero vector)
         */
        constexpr float2() noexcept : x(0.0f), y(0.0f) {}

        /**
         * @brief Construct from components
         * @param x X component
         * @param y Y component
         */
        constexpr float2(float x, float y) noexcept : x(x), y(y) {}

        /**
         * @brief Construct from scalar (all components set to same value)
         * @param scalar Value for all components
         */
        explicit constexpr float2(float scalar) noexcept : x(scalar), y(scalar) {}

        /**
         * @brief Copy constructor
         */
        constexpr float2(const float2&) noexcept = default;

        /**
         * @brief Construct from raw float array
         * @param data Pointer to float array [x, y]
         */
        explicit float2(const float* data) noexcept;

        /**
         * @brief Construct from SSE register (advanced users)
         * @param simd_ SSE register containing vector data
         */
        explicit float2(__m128 simd_) noexcept;

#if defined(MATH_SUPPORT_D3DX)
        /**
         * @brief Construct from D3DXVECTOR2
         * @param vec DirectX 2D vector
         */
        float2(const D3DXVECTOR2& vec) noexcept;

        /**
         * @brief Construct from D3DXVECTOR4 (extracts x, y)
         * @param vec DirectX 4D vector
         */
        float2(const D3DXVECTOR4& vec) noexcept;

        /**
         * @brief Construct from D3DCOLOR (RGB to grayscale)
         * @param color DirectX color value
         */
        explicit float2(D3DCOLOR color) noexcept;
#endif

        // ============================================================================
        // Assignment Operators
        // ============================================================================

        /**
         * @brief Copy assignment operator
         */
        float2& operator=(const float2&) noexcept = default;

        /**
         * @brief Scalar assignment (sets all components to same value)
         * @param scalar Value for all components
         */
        float2& operator=(float scalar) noexcept;

#if defined(MATH_SUPPORT_D3DX)
        /**
         * @brief Assignment from D3DXVECTOR2
         * @param vec DirectX 2D vector
         */
        float2& operator=(const D3DXVECTOR2& vec) noexcept;

        /**
         * @brief Assignment from D3DCOLOR
         * @param color DirectX color value
         */
        float2& operator=(D3DCOLOR color) noexcept;
#endif

        // ============================================================================
        // Compound Assignment Operators
        // ============================================================================

        float2& operator+=(const float2& rhs) noexcept;
        float2& operator-=(const float2& rhs) noexcept;
        float2& operator*=(const float2& rhs) noexcept;
        float2& operator/=(const float2& rhs) noexcept;
        float2& operator*=(float scalar) noexcept;
        float2& operator/=(float scalar) noexcept;

        float2 operator+(const float2& rhs) const noexcept;
        float2 operator-(const float2& rhs) const noexcept;
        float2 operator+(const float& rhs) const noexcept;
        float2 operator-(const float& rhs) const noexcept;

        // ============================================================================
        // Unary Operators
        // ============================================================================

        constexpr float2 operator+() const noexcept { return *this; }
        constexpr float2 operator-() const noexcept { return float2(-x, -y); }

        // ============================================================================
        // Access Operators
        // ============================================================================

        /**
         * @brief Access component by index
         * @param index Component index (0 = x, 1 = y)
         * @return Reference to component
         */
        float& operator[](int index) noexcept;

        /**
         * @brief Access component by index (const)
         */
        const float& operator[](int index) const noexcept;

        // ============================================================================
        // Conversion Operators
        // ============================================================================

        /**
         * @brief Convert to const float pointer (for interoperability)
         */
        operator const float* () const noexcept;

        /**
         * @brief Convert to float pointer (for interoperability)
         */
        operator float* () noexcept;

        /**
         * @brief Convert to SSE register (advanced users)
         */
        operator __m128() const noexcept;

#if defined(MATH_SUPPORT_D3DX)
        /**
         * @brief Convert to D3DXVECTOR2
         */
        operator D3DXVECTOR2() const noexcept;
#endif

        // ============================================================================
        // Static Constructors
        // ============================================================================

        /**
         * @brief Zero vector (0, 0)
         */
        static constexpr float2 zero() noexcept { return float2(0.0f, 0.0f); }

        /**
         * @brief One vector (1, 1)
         */
        static constexpr float2 one() noexcept { return float2(1.0f, 1.0f); }

        /**
         * @brief Unit X vector (1, 0)
         */
        static constexpr float2 unit_x() noexcept { return float2(1.0f, 0.0f); }

        /**
         * @brief Unit Y vector (0, 1)
         */
        static constexpr float2 unit_y() noexcept { return float2(0.0f, 1.0f); }

        // ============================================================================
        // Mathematical Functions
        // ============================================================================

        /**
         * @brief Compute Euclidean length (magnitude)
         * @return Length of the vector
         */
        float length() const noexcept;

        /**
         * @brief Compute squared length (faster, useful for comparisons)
         * @return Squared length of the vector
         */
        constexpr float length_sq() const noexcept { return x * x + y * y; }

        /**
         * @brief Normalize vector to unit length
         * @return Normalized vector
         * @note Returns zero vector if length is zero
         */
        float2 normalize() const noexcept;

        /**
         * @brief Compute dot product with another vector
         * @param other Other vector
         * @return Dot product result
         */
        float dot(const float2& other) const noexcept {
            __m128 va = _mm_set_ps(0, 0, y, x);
            __m128 vb = _mm_set_ps(0, 0, other.y, other.x);
            __m128 product = _mm_mul_ps(va, vb);
            __m128 sum = _mm_hadd_ps(product, product);
            return _mm_cvtss_f32(sum);
        }

        /**
         * @brief Compute 2D cross product (scalar result)
         * @param other Other vector
         * @return Cross product result (x1*y2 - y1*x2)
         */
        float cross(const float2& other) 
        {
            return _mm_cvtss_f32(_mm_sub_ss(
                _mm_mul_ss(_mm_set_ss(x), _mm_set_ss(other.y)),
                _mm_mul_ss(_mm_set_ss(y), _mm_set_ss(other.x))
            ));
        }

        /**
         * @brief Compute distance to another point
         * @param other Other point
         * @return Euclidean distance
         */
        float distance(const float2& other) const noexcept {
            float dx = x - other.x;
            float dy = y - other.y;
            return std::sqrt(dx * dx + dy * dy);
        }

        /**
         * @brief Compute squared distance to another point (faster)
         * @param other Other point
         * @return Squared Euclidean distance
         */
        constexpr float distance_sq(const float2& other) const noexcept {
            float dx = x - other.x;
            float dy = y - other.y;
            return dx * dx + dy * dy;
        }

        // ============================================================================
        // HLSL-like Functions
        // ============================================================================

        /**
         * @brief HLSL-like abs function (component-wise absolute value)
         * @return Vector with absolute values of components
         */
        float2 abs() const noexcept;

        /**
         * @brief HLSL-like sign function (component-wise sign)
         * @return Vector with signs of components (-1, 0, or 1)
         */
        float2 sign() const noexcept;

        /**
         * @brief HLSL-like floor function (component-wise floor)
         * @return Vector with floored components
         */
        float2 floor() const noexcept;

        /**
         * @brief HLSL-like ceil function (component-wise ceiling)
         * @return Vector with ceiling components
         */
        float2 ceil() const noexcept;

        /**
         * @brief HLSL-like round function (component-wise rounding)
         * @return Vector with rounded components
         */
        float2 round() const noexcept;

        /**
         * @brief HLSL-like frac function (component-wise fractional part)
         * @return Vector with fractional parts of components
         */
        float2 frac() const noexcept;

        /**
         * @brief HLSL-like saturate function (clamp components to [0, 1])
         * @return Saturated vector
         */
        float2 saturate() const noexcept;

        /**
         * @brief HLSL-like step function (component-wise step)
         * @param edge Edge value
         * @return 1.0 if component >= edge, else 0.0
         */
        float2 step(float edge) const noexcept;

        /**
         * @brief HLSL-like smoothstep function (component-wise smooth interpolation)
         * @param edge0 Lower edge
         * @param edge1 Upper edge
         * @return Smoothly interpolated vector
         */
        float2 smoothstep(float edge0, float edge1) const noexcept;

        // ============================================================================
        // Geometric Operations
        // ============================================================================

        /**
         * @brief Compute perpendicular vector (90 degree counter-clockwise rotation)
         * @return Perpendicular vector (-y, x)
         */
        constexpr float2 perpendicular() const noexcept { return float2(-y, x); }

        /**
         * @brief Compute reflection vector
         * @param normal Surface normal (must be normalized)
         * @return Reflected vector
         */
        float2 reflect(const float2& normal) const noexcept;

        /**
         * @brief Compute refraction vector
         * @param normal Surface normal (must be normalized)
         * @param eta Ratio of indices of refraction
         * @return Refracted vector
         */
        float2 refract(const float2& normal, float eta) const noexcept;

        /**
         * @brief Rotate vector by angle (radians)
         * @param angle Rotation angle in radians (positive = counter-clockwise)
         * @return Rotated vector
         */
        float2 rotate(float angle) const noexcept;

        /**
         * @brief Compute angle of this vector relative to X-axis
         * @return Angle in radians between [-π, π]
         */
        float angle() const noexcept;

        // ============================================================================
        // Static Mathematical Functions
        // ============================================================================

        /**
         * @brief Compute dot product of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Dot product result
         */
        static float dot(const float2& a, const float2& b) noexcept {
            __m128 va = _mm_set_ps(0, 0, a.y, a.x);
            __m128 vb = _mm_set_ps(0, 0, b.y, b.x);
            __m128 product = _mm_mul_ps(va, vb);
            __m128 sum = _mm_hadd_ps(product, product);
            return _mm_cvtss_f32(sum);
        }

        /**
         * @brief Compute cross product of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Cross product result
         */
        static float cross(const float2& a, const float2& b) noexcept {
            return _mm_cvtss_f32(_mm_sub_ss(
                _mm_mul_ss(_mm_set_ss(a.x), _mm_set_ss(b.y)),
                _mm_mul_ss(_mm_set_ss(a.y), _mm_set_ss(b.x))
            ));
        }

        /**
         * @brief Linear interpolation between two vectors
         * @param a Start vector
         * @param b End vector
         * @param t Interpolation factor [0, 1]
         * @return Interpolated vector
         */
        static constexpr float2 lerp(const float2& a, const float2& b, float t) noexcept {
            return float2(
                a.x + (b.x - a.x) * t,
                a.y + (b.y - a.y) * t
            );
        }

        /**
         * @brief Spherical linear interpolation (for directions)
         * @param a Start vector (should be normalized)
         * @param b End vector (should be normalized)
         * @param t Interpolation factor [0, 1]
         * @return Interpolated vector
         */
        static float2 slerp(const float2& a, const float2& b, float t) noexcept;

        /**
         * @brief Component-wise minimum of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Component-wise minimum
         */
        static constexpr float2 min(const float2& a, const float2& b) noexcept {
            return float2(
                (a.x < b.x) ? a.x : b.x,
                (a.y < b.y) ? a.y : b.y
            );
        }

        /**
         * @brief Component-wise maximum of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Component-wise maximum
         */
        static constexpr float2 max(const float2& a, const float2& b) noexcept {
            return float2(
                (a.x > b.x) ? a.x : b.x,
                (a.y > b.y) ? a.y : b.y
            );
        }

        // ============================================================================
        // Swizzle Operations
        // ============================================================================

        /**
         * @brief Swizzle to (y, x)
         * @return Vector with components swapped
         */
        constexpr float2 yx() const noexcept { return float2(y, x); }

        /**
         * @brief Swizzle to (x, x)
         * @return Vector with x component duplicated
         */
        constexpr float2 xx() const noexcept { return float2(x, x); }

        /**
         * @brief Swizzle to (y, y)
         * @return Vector with y component duplicated
         */
        constexpr float2 yy() const noexcept { return float2(y, y); }

        // ============================================================================
        // Utility Methods
        // ============================================================================

        /**
         * @brief Check if vector contains finite values
         * @return True if all components are finite (not NaN or infinity)
         */
        bool isValid() const noexcept;

        /**
         * @brief Check if vector is approximately equal to another
         * @param other Vector to compare with
         * @param epsilon Comparison tolerance
         * @return True if vectors are approximately equal
         */
        bool approximately(const float2& other, float epsilon = EPSILON) const noexcept;

        /**
         * @brief Check if vector is approximately zero
         * @param epsilon Comparison tolerance
         * @return True if vector length is approximately zero
         */
        bool approximately_zero(float epsilon = EPSILON) const noexcept;

        /**
         * @brief Check if vector is normalized
         * @param epsilon Comparison tolerance
         * @return True if vector length is approximately 1.0
         */
        bool is_normalized(float epsilon = EPSILON) const noexcept;

        /**
         * @brief Convert to string representation
         * @return String in format "(x, y)"
         */
        std::string to_string() const;

        /**
         * @brief Get pointer to raw data
         * @return Pointer to first component
         */
        const float* data() const noexcept { return &x; }

        /**
         * @brief Get pointer to raw data (mutable)
         * @return Pointer to first component
         */
        float* data() noexcept { return &x; }

        // ============================================================================
        // Comparison Operators
        // ============================================================================

        bool operator==(const float2& rhs) const noexcept;
        bool operator!=(const float2& rhs) const noexcept;
    };

    //============================================================================
    // Binary Operators
    // ============================================================================

    /**
     * @brief Component-wise vector multiplication
     * @param lhs Left-hand side vector
     * @param rhs Right-hand side vector
     * @return Result of multiplication
     */
    inline float2 operator*(float2 lhs, const float2& rhs) noexcept {
        return lhs *= rhs;
    }

    /**
     * @brief Component-wise vector division
     * @param lhs Left-hand side vector
     * @param rhs Right-hand side vector
     * @return Result of division
     */
    inline float2 operator/(float2 lhs, const float2& rhs) noexcept {
        return lhs /= rhs;
    }

    /**
     * @brief Vector-scalar multiplication
     * @param vec Vector to multiply
     * @param scalar Scalar multiplier
     * @return Scaled vector
     */
    inline float2 operator*(float2 vec, float scalar) noexcept {
        return vec *= scalar;
    }

    /**
     * @brief Scalar-vector multiplication
     * @param scalar Scalar multiplier
     * @param vec Vector to multiply
     * @return Scaled vector
     */
    inline float2 operator*(float scalar, float2 vec) noexcept {
        return vec *= scalar;
    }

    /**
     * @brief Vector-scalar division
     * @param vec Vector to divide
     * @param scalar Scalar divisor
     * @return Scaled vector
     */
    inline float2 operator/(float2 vec, float scalar) noexcept {
        return vec /= scalar;
    }

    inline float2 operator+(float scalar, float2 vec) noexcept {
        return vec + scalar;
    }

    // ============================================================================
    // Global Mathematical Functions
    // ============================================================================

    /**
     * @brief Compute distance between two points
     * @param a First point
     * @param b Second point
     * @return Euclidean distance between points
     */
    inline float distance(const float2& a, const float2& b) noexcept {
        return (b - a).length();
    }

    /**
     * @brief Compute squared distance between two points (faster)
     * @param a First point
     * @param b Second point
     * @return Squared Euclidean distance
     */
    inline float distance_sq(const float2& a, const float2& b) noexcept {
        return (b - a).length_sq();
    }

    /**
     * @brief Compute dot product of two vectors
     * @param a First vector
     * @param b Second vector
     * @return Dot product result
     */
    inline float dot(const float2& a, const float2& b) noexcept {
        return a.dot(b);
    }

    /**
     * @brief Compute cross product of two vectors
     * @param a First vector
     * @param b Second vector
     * @return Cross product result
     */
    inline float cross(const float2& a, const float2& b) noexcept {
        return float2::cross(a, b);
    }

    /**
     * @brief Check if two vectors are approximately equal
     * @param a First vector
     * @param b Second vector
     * @param epsilon Comparison tolerance
     * @return True if vectors are approximately equal
     */
    inline bool approximately(const float2& a, const float2& b, float epsilon) noexcept {
        return a.approximately(b, epsilon);
    }

    /**
     * @brief Check if vector contains valid finite values
     * @param vec Vector to check
     * @return True if vector is valid (finite values)
     */
    inline bool isValid(const float2& vec) noexcept {
        return vec.isValid();
    }

    /**
     * @brief Linear interpolation between two vectors
     * @param a Start vector
     * @param b End vector
     * @param t Interpolation factor [0, 1]
     * @return Interpolated vector
     */
    inline float2 lerp(const float2& a, const float2& b, float t) noexcept {
        return float2::lerp(a, b, t);
    }

    /**
     * @brief Spherical linear interpolation between two vectors
     * @param a Start vector (should be normalized)
     * @param b End vector (should be normalized)
     * @param t Interpolation factor [0, 1]
     * @return Interpolated vector
     */
    inline float2 slerp(const float2& a, const float2& b, float t) noexcept {
        return float2::slerp(a, b, t);
    }

    // ============================================================================
    // Geometric Operations
    // ============================================================================

    /**
     * @brief Compute perpendicular vector (90 degree counter-clockwise rotation)
     * @param vec Input vector
     * @return Perpendicular vector (-y, x)
     */
    inline float2 perpendicular(const float2& vec) noexcept {
        return vec.perpendicular();
    }

    /**
     * @brief Compute reflection vector
     * @param incident Incident vector
     * @param normal Surface normal (must be normalized)
     * @return Reflected vector
     */
    inline float2 reflect(const float2& incident, const float2& normal) noexcept {
        return incident.reflect(normal);
    }

    /**
     * @brief Compute refraction vector
     * @param incident Incident vector
     * @param normal Surface normal (must be normalized)
     * @param eta Ratio of indices of refraction
     * @return Refracted vector
     */
    inline float2 refract(const float2& incident, const float2& normal, float eta) noexcept {
        return incident.refract(normal, eta);
    }

    /**
     * @brief Rotate vector by angle (radians)
     * @param vec Vector to rotate
     * @param angle Rotation angle in radians (positive = counter-clockwise)
     * @return Rotated vector
     */
    inline float2 rotate(const float2& vec, float angle) noexcept {
        return vec.rotate(angle);
    }

    /**
     * @brief Compute angle between two vectors in radians
     * @param a First vector
     * @param b Second vector
     * @return Angle in radians between [0, π]
     */
    inline float angle_between(const float2& a, const float2& b) noexcept {
        float2 a_norm = a.normalize();
        float2 b_norm = b.normalize();

        float dot_val = dot(a_norm, b_norm);
        dot_val = MathFunctions::clamp(dot_val, -1.0f, 1.0f);

        return std::acos(dot_val);
    }

    /**
     * @brief Compute signed angle between two vectors in radians
     * @param from Starting vector
     * @param to Target vector
     * @return Signed angle in radians between [-π, π]
     */
    inline float signed_angle_between(const float2& from, const float2& to) noexcept {
        float2 from_norm = from.normalize();
        float2 to_norm = to.normalize();

        float dot_val = dot(from_norm, to_norm);
        dot_val = MathFunctions::clamp(dot_val, -1.0f, 1.0f);
        float angle = std::acos(dot_val);

        float cross_val = cross(from_norm, to_norm);
        return (cross_val < 0.0f) ? -angle : angle;
    }

    /**
     * @brief Project vector onto another vector
     * @param vec Vector to project
     * @param onto Vector to project onto
     * @return Projected vector
     */
    inline float2 project(const float2& vec, const float2& onto) noexcept {
        float onto_length_sq = onto.length_sq();

        if (onto_length_sq < Constants::Constants<float>::Epsilon) {
            return float2::zero();
        }

        float dot_val = dot(vec, onto);
        return onto * (dot_val / onto_length_sq);
    }

    /**
     * @brief Reject vector from another vector (component perpendicular)
     * @param vec Vector to reject
     * @param from Vector to reject from
     * @return Rejected vector
     */
    inline float2 reject(const float2& vec, const float2& from) noexcept {
        return vec - project(vec, from);
    }

    // ============================================================================
    // HLSL-like Global Functions
    // ============================================================================

    /**
     * @brief HLSL-like abs function (component-wise absolute value)
     * @param vec Input vector
     * @return Vector with absolute values of components
     */
    inline float2 abs(const float2& vec) noexcept {
        return vec.abs();
    }

    /**
     * @brief HLSL-like sign function (component-wise sign)
     * @param vec Input vector
     * @return Vector with signs of components
     */
    inline float2 sign(const float2& vec) noexcept {
        return vec.sign();
    }

    /**
     * @brief HLSL-like floor function (component-wise floor)
     * @param vec Input vector
     * @return Vector with floored components
     */
    inline float2 floor(const float2& vec) noexcept {
        return vec.floor();
    }

    /**
     * @brief HLSL-like ceil function (component-wise ceiling)
     * @param vec Input vector
     * @return Vector with ceiling components
     */
    inline float2 ceil(const float2& vec) noexcept {
        return vec.ceil();
    }

    /**
     * @brief HLSL-like round function (component-wise rounding)
     * @param vec Input vector
     * @return Vector with rounded components
     */
    inline float2 round(const float2& vec) noexcept {
        return vec.round();
    }

    /**
     * @brief HLSL-like frac function (component-wise fractional part)
     * @param vec Input vector
     * @return Vector with fractional parts of components
     */
    inline float2 frac(const float2& vec) noexcept {
        return vec.frac();
    }

    /**
     * @brief HLSL-like saturate function (clamp components to [0, 1])
     * @param vec Input vector
     * @return Saturated vector
     */
    inline float2 saturate(const float2& vec) noexcept {
        return vec.saturate();
    }

    /**
     * @brief HLSL-like step function (component-wise step)
     * @param edge Edge value
     * @param vec Input vector
     * @return Step result vector
     */
    inline float2 step(float edge, const float2& vec) noexcept {
        return vec.step(edge);
    }

    /**
     * @brief HLSL-like smoothstep function (component-wise smooth interpolation)
     * @param edge0 Lower edge
     * @param edge1 Upper edge
     * @param vec Input vector
     * @return Smoothly interpolated vector
     */
    inline float2 smoothstep(float edge0, float edge1, const float2& vec) noexcept {
        return vec.smoothstep(edge0, edge1);
    }

    /**
     * @brief HLSL-like clamp function (component-wise clamping)
     * @param vec Vector to clamp
     * @param min_val Minimum values
     * @param max_val Maximum values
     * @return Clamped vector
     */
    inline float2 clamp(const float2& vec, const float2& min_val, const float2& max_val) noexcept {
        return float2(
            MathFunctions::clamp(vec.x, min_val.x, max_val.x),
            MathFunctions::clamp(vec.y, min_val.y, max_val.y)
        );
    }

    /**
     * @brief HLSL-like min function (component-wise minimum)
     * @param a First vector
     * @param b Second vector
     * @return Component-wise minimum
     */
    inline float2 min(const float2& a, const float2& b) noexcept {
        return float2::min(a, b);
    }

    /**
     * @brief HLSL-like max function (component-wise maximum)
     * @param a First vector
     * @param b Second vector
     * @return Component-wise maximum
     */
    inline float2 max(const float2& a, const float2& b) noexcept {
        return float2::max(a, b);
    }

    // ============================================================================
    // D3D Compatibility Functions
    // ============================================================================

#if defined(MATH_SUPPORT_D3DX)

    /**
     * @brief Convert float2 to D3DXVECTOR2
     * @param vec float2 vector to convert
     * @return D3DXVECTOR2 equivalent
     */
    inline D3DXVECTOR2 ToD3DXVECTOR2(const float2& vec) noexcept {
        return D3DXVECTOR2(vec.x, vec.y);
    }

    /**
     * @brief Convert D3DXVECTOR2 to float2
     * @param vec D3DXVECTOR2 to convert
     * @return float2 equivalent
     */
    inline float2 FromD3DXVECTOR2(const D3DXVECTOR2& vec) noexcept {
        return float2(vec.x, vec.y);
    }

    /**
     * @brief Convert float2 to D3DCOLOR (uses x,y as R,G channels)
     * @param color float2 representing color (x=R, y=G)
     * @return D3DCOLOR equivalent
     */
    inline D3DCOLOR ToD3DCOLOR(const float2& color) noexcept {
        return D3DCOLOR_COLORVALUE(color.x, color.y, 0.0f, 1.0f);
    }

    /**
     * @brief Convert array of float2 to array of D3DXVECTOR2
     * @param source Source float2 array
     * @param destination Destination D3DXVECTOR2 array
     * @param count Number of elements to convert
     */
    inline void float2ArrayToD3D(const float2* source, D3DXVECTOR2* destination, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            destination[i] = ToD3DXVECTOR2(source[i]);
        }
    }

    /**
     * @brief Convert array of D3DXVECTOR2 to array of float2
     * @param source Source D3DXVECTOR2 array
     * @param destination Destination float2 array
     * @param count Number of elements to convert
     */
    inline void D3DArrayTofloat2(const D3DXVECTOR2* source, float2* destination, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            destination[i] = FromD3DXVECTOR2(source[i]);
        }
    }

#endif // MATH_SUPPORT_D3DX

    // ============================================================================
    // Additional Utility Functions
    // ============================================================================

    /**
     * @brief Compute distance from point to line segment
     * @param point The point
     * @param line_start Start of line segment
     * @param line_end End of line segment
     * @return Shortest distance from point to line segment
     */
    inline float distance_to_line_segment(const float2& point,
        const float2& line_start,
        const float2& line_end) noexcept {
        float2 line_vec = line_end - line_start;
        float2 point_vec = point - line_start;

        float line_length_sq = line_vec.length_sq();

        if (line_length_sq < Constants::Constants<float>::Epsilon) {
            return point_vec.length();
        }

        float t = dot(point_vec, line_vec) / line_length_sq;
        t = MathFunctions::clamp(t, 0.0f, 1.0f);

        float2 closest_point = line_start + line_vec * t;
        return distance(point, closest_point);
    }

    /**
     * @brief Check if point is inside triangle
     * @param point Point to check
     * @param a First triangle vertex
     * @param b Second triangle vertex
     * @param c Third triangle vertex
     * @return True if point is inside triangle
     */
    inline bool point_in_triangle(const float2& point,
        const float2& a,
        const float2& b,
        const float2& c) noexcept {
        // Compute vectors
        float2 v0 = c - a;
        float2 v1 = b - a;
        float2 v2 = point - a;

        // Compute dot products
        float dot00 = dot(v0, v0);
        float dot01 = dot(v0, v1);
        float dot02 = dot(v0, v2);
        float dot11 = dot(v1, v1);
        float dot12 = dot(v1, v2);

        // Compute barycentric coordinates
        float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01);
        float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

        // Check if point is in triangle
        return (u >= 0.0f) && (v >= 0.0f) && (u + v <= 1.0f);
    }

    // ============================================================================
    // Useful Constants
    // ============================================================================

    /**
     * @brief Zero vector constant (0, 0)
     */
    extern const float2 float2_Zero;

    /**
     * @brief One vector constant (1, 1)
     */
    extern const float2 float2_One;

    /**
     * @brief Unit X vector constant (1, 0)
     */
    extern const float2 float2_UnitX;

    /**
     * @brief Unit Y vector constant (0, 1)
     */
    extern const float2 float2_UnitY;

    /**
     * @brief Right vector constant (1, 0) - alias for UnitX
     */
    extern const float2 float2_Right;

    /**
     * @brief Left vector constant (-1, 0)
     */
    extern const float2 float2_Left;

    /**
     * @brief Up vector constant (0, 1) - alias for UnitY
     */
    extern const float2 float2_Up;

    /**
     * @brief Down vector constant (0, -1)
     */
    extern const float2 float2_Down;

} // namespace Math
