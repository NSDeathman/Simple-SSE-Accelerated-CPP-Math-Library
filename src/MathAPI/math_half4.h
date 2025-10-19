// Description: 4-dimensional half-precision vector class with 
//              comprehensive mathematical operations, SSE optimization,
//              and full HLSL compatibility
// Author: NSDeathman, DeepSeek
#pragma once

/**
 * @file math_half4.h
 * @brief 4-dimensional half-precision vector class
 * @note Optimized for 4D graphics, homogeneous coordinates, RGBA colors with SSE optimization
 * @note Features comprehensive HLSL compatibility and color space operations
 */

#include <xmmintrin.h>
#include <pmmintrin.h> // SSE3
#include <smmintrin.h> // SSE4.1
#include <cmath>
#include <algorithm>
#include <string>
#include <cstdio>

#include "math_config.h"
#include "math_constants.h"
#include "math_functions.h"
#include "math_half.h"
#include "math_half2.h"
#include "math_half3.h"
#include "math_float4.h"

 // Platform-specific support
#if defined(MATH_SUPPORT_D3DX)
#include <d3dx9.h>  // D3DXVECTOR4, D3DXVECTOR3, D3DXVECTOR2, D3DCOLOR
#endif

namespace Math
{
    /**
     * @class half4
     * @brief 4-dimensional half-precision vector with comprehensive mathematical operations
     *
     * Represents a 4D vector (x, y, z, w) using 16-bit half-precision floating point format.
     * Features SSE optimization for performance-critical operations and comprehensive
     * HLSL compatibility. Perfect for 4D graphics, homogeneous coordinates, RGBA colors,
     * and memory-constrained applications where full 32-bit precision is not required.
     *
     * @note Optimized for memory bandwidth and GPU data formats
     * @note Provides seamless interoperability with float4 and comprehensive mathematical operations
     * @note Includes advanced color operations, geometric functions, and homogeneous coordinate support
     */
    class MATH_API half4
    {
    public:
        // ============================================================================
        // Data Members (Public for Direct Access)
        // ============================================================================

        half x; ///< X component of the vector
        half y; ///< Y component of the vector
        half z; ///< Z component of the vector
        half w; ///< W component of the vector (homogeneous coordinate or alpha)

        // ============================================================================
        // Constructors
        // ============================================================================

        /**
         * @brief Default constructor (initializes to zero vector)
         */
        half4() noexcept;

        /**
         * @brief Construct from half components
         * @param x X component
         * @param y Y component
         * @param z Z component
         * @param w W component
         */
        half4(half x, half y, half z, half w) noexcept;

        /**
         * @brief Construct from scalar (all components set to same value)
         * @param scalar Value for all components
         */
        explicit half4(half scalar) noexcept;

        /**
         * @brief Construct from float components
         * @param x X component as float
         * @param y Y component as float
         * @param z Z component as float
         * @param w W component as float
         */
        half4(float x, float y, float z, float w) noexcept;

        /**
         * @brief Construct from float scalar (all components set to same value)
         * @param scalar Value for all components as float
         */
        explicit half4(float scalar) noexcept;

        /**
         * @brief Copy constructor
         */
        half4(const half4&) noexcept = default;

        /**
         * @brief Construct from half2 and z, w components
         * @param vec 2D vector for x and y components
         * @param z Z component
         * @param w W component
         */
        half4(const half2& vec, half z = half::from_bits(0), half w = half::from_bits(0)) noexcept;

        /**
         * @brief Construct from half3 and w component
         * @param vec 3D vector for x, y, z components
         * @param w W component
         */
        half4(const half3& vec, half w = half::from_bits(0)) noexcept;

        /**
         * @brief Construct from float4 (converts components to half precision)
         * @param vec 32-bit floating point vector
         */
        half4(const float4& vec) noexcept;

        /**
         * @brief Construct from float2 and z, w components
         * @param vec 2D vector for x and y components
         * @param z Z component as float
         * @param w W component as float
         */
        half4(const float2& vec, float z = 0.0f, float w = 0.0f) noexcept;

        /**
         * @brief Construct from float3 and w component
         * @param vec 3D vector for x, y, z components
         * @param w W component as float
         */
        half4(const float3& vec, float w = 0.0f) noexcept;

        // ============================================================================
        // Assignment Operators
        // ============================================================================

        /**
         * @brief Copy assignment operator
         */
        half4& operator=(const half4&) noexcept = default;

        /**
         * @brief Assignment from float4 (converts components to half precision)
         * @param vec 32-bit floating point vector
         */
        half4& operator=(const float4& vec) noexcept;

        /**
         * @brief Assignment from half3 (preserves w component)
         * @param xyz 3D vector for x, y, z components
         */
        half4& operator=(const half3& xyz) noexcept;

        /**
         * @brief Assignment from half scalar (sets all components to same value)
         * @param scalar Value for all components
         */
        half4& operator=(half scalar) noexcept;

        /**
         * @brief Assignment from float scalar (sets all components to same value)
         * @param scalar Value for all components as float
         */
        half4& operator=(float scalar) noexcept;

        // ============================================================================
        // Compound Assignment Operators
        // ============================================================================

        /**
         * @brief Compound addition assignment
         * @param rhs Vector to add
         * @return Reference to this object
         */
        half4& operator+=(const half4& rhs) noexcept;

        /**
         * @brief Compound subtraction assignment
         * @param rhs Vector to subtract
         * @return Reference to this object
         */
        half4& operator-=(const half4& rhs) noexcept;

        /**
         * @brief Compound multiplication assignment
         * @param rhs Vector to multiply by
         * @return Reference to this object
         */
        half4& operator*=(const half4& rhs) noexcept;

        /**
         * @brief Compound division assignment
         * @param rhs Vector to divide by
         * @return Reference to this object
         */
        half4& operator/=(const half4& rhs) noexcept;

        /**
         * @brief Compound scalar multiplication assignment (half)
         * @param scalar Scalar to multiply by
         * @return Reference to this object
         */
        half4& operator*=(half scalar) noexcept;

        /**
         * @brief Compound scalar multiplication assignment (float)
         * @param scalar Scalar to multiply by
         * @return Reference to this object
         */
        half4& operator*=(float scalar) noexcept;

        /**
         * @brief Compound scalar division assignment (half)
         * @param scalar Scalar to divide by
         * @return Reference to this object
         */
        half4& operator/=(half scalar) noexcept;

        /**
         * @brief Compound scalar division assignment (float)
         * @param scalar Scalar to divide by
         * @return Reference to this object
         */
        half4& operator/=(float scalar) noexcept;

        // ============================================================================
        // Unary Operators
        // ============================================================================

        /**
         * @brief Unary plus operator
         * @return Positive vector
         */
        half4 operator+() const noexcept;

        /**
         * @brief Unary minus operator
         * @return Negated vector
         */
        half4 operator-() const noexcept;

        // ============================================================================
        // Access Operators
        // ============================================================================

        /**
         * @brief Access component by index
         * @param index Component index (0 = x, 1 = y, 2 = z, 3 = w)
         * @return Reference to component
         */
        half& operator[](int index) noexcept;

        /**
         * @brief Access component by index (const)
         * @param index Component index (0 = x, 1 = y, 2 = z, 3 = w)
         * @return Const reference to component
         */
        const half& operator[](int index) const noexcept;

        // ============================================================================
        // Conversion Operators
        // ============================================================================

        /**
         * @brief Convert to float4 (promotes components to full precision)
         * @return 32-bit floating point vector
         */
        explicit operator float4() const noexcept;

        // ============================================================================
        // Static Constructors
        // ============================================================================

        /**
         * @brief Zero vector (0, 0, 0, 0)
         * @return Zero vector
         */
        static half4 zero() noexcept;

        /**
         * @brief One vector (1, 1, 1, 1)
         * @return One vector
         */
        static half4 one() noexcept;

        /**
         * @brief Unit X vector (1, 0, 0, 0)
         * @return Unit X vector
         */
        static half4 unit_x() noexcept;

        /**
         * @brief Unit Y vector (0, 1, 0, 0)
         * @return Unit Y vector
         */
        static half4 unit_y() noexcept;

        /**
         * @brief Unit Z vector (0, 0, 1, 0)
         * @return Unit Z vector
         */
        static half4 unit_z() noexcept;

        /**
         * @brief Unit W vector (0, 0, 0, 1)
         * @return Unit W vector
         */
        static half4 unit_w() noexcept;

        /**
         * @brief Create vector from RGBA color values [0-255]
         * @param r Red component [0-255]
         * @param g Green component [0-255]
         * @param b Blue component [0-255]
         * @param a Alpha component [0-255] (default: 255)
         * @return RGBA color vector
         */
        static half4 from_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) noexcept;

        // ============================================================================
        // Mathematical Functions (SSE Optimized)
        // ============================================================================

        /**
         * @brief Compute Euclidean length (magnitude)
         * @return Length of the vector
         */
        half length() const noexcept;

        /**
         * @brief Compute squared length (faster, useful for comparisons)
         * @return Squared length of the vector
         */
        half length_sq() const noexcept;

        /**
         * @brief Normalize vector to unit length
         * @return Normalized vector
         * @note Returns zero vector if length is zero
         */
        half4 normalize() const noexcept;

        /**
         * @brief Compute dot product with another vector
         * @param other Other vector
         * @return Dot product result
         */
        half dot(const half4& other) const noexcept;

        /**
         * @brief Compute 3D dot product (ignores w component)
         * @param other Other vector
         * @return 3D dot product result
         */
        half dot3(const half4& other) const noexcept;

        /**
         * @brief Compute 3D cross product (ignores w component)
         * @param other Other vector
         * @return 3D cross product result (w = 0)
         */
        half4 cross(const half4& other) const noexcept;

        /**
         * @brief Compute distance to another point
         * @param other Other point
         * @return Euclidean distance
         */
        half distance(const half4& other) const noexcept;

        /**
         * @brief Compute squared distance to another point (faster)
         * @param other Other point
         * @return Squared Euclidean distance
         */
        half distance_sq(const half4& other) const noexcept;

        // ============================================================================
        // HLSL-like Functions
        // ============================================================================

        /**
         * @brief HLSL-like abs function (component-wise absolute value)
         * @return Vector with absolute values of components
         */
        half4 abs() const noexcept;

        /**
         * @brief HLSL-like sign function (component-wise sign)
         * @return Vector with signs of components (-1, 0, or 1)
         */
        half4 sign() const noexcept;

        /**
         * @brief HLSL-like floor function (component-wise floor)
         * @return Vector with floored components
         */
        half4 floor() const noexcept;

        /**
         * @brief HLSL-like ceil function (component-wise ceiling)
         * @return Vector with ceiling components
         */
        half4 ceil() const noexcept;

        /**
         * @brief HLSL-like round function (component-wise rounding)
         * @return Vector with rounded components
         */
        half4 round() const noexcept;

        /**
         * @brief HLSL-like frac function (component-wise fractional part)
         * @return Vector with fractional parts of components
         */
        half4 frac() const noexcept;

        /**
         * @brief HLSL-like saturate function (clamp components to [0, 1])
         * @return Saturated vector
         */
        half4 saturate() const noexcept;

        /**
         * @brief HLSL-like step function (component-wise step)
         * @param edge Edge value
         * @return 1.0 if component >= edge, else 0.0
         */
        half4 step(half edge) const noexcept;

        // ============================================================================
        // Color Operations
        // ============================================================================

        /**
         * @brief Compute luminance using Rec. 709 weights (ignores alpha)
         * @return Luminance value
         * @note Uses weights: 0.2126*R + 0.7152*G + 0.0722*B
         */
        half luminance() const noexcept;

        /**
         * @brief Compute average brightness (simple average of RGB, ignores alpha)
         * @return Brightness value
         */
        half brightness() const noexcept;

        /**
         * @brief Premultiply RGB components by alpha
         * @return Premultiplied color
         */
        half4 premultiply_alpha() const noexcept;

        /**
         * @brief Unpremultiply RGB components (divide by alpha)
         * @return Unpremultiplied color
         * @note Returns original color if alpha is zero
         */
        half4 unpremultiply_alpha() const noexcept;

        /**
         * @brief Convert to grayscale using luminance
         * @return Grayscale color (RGB = luminance, alpha preserved)
         */
        half4 grayscale() const noexcept;

        /**
         * @brief Apply sRGB to linear conversion
         * @return Linear color values
         */
        half4 srgb_to_linear() const noexcept;

        /**
         * @brief Apply linear to sRGB conversion
         * @return sRGB color values
         */
        half4 linear_to_srgb() const noexcept;

        // ============================================================================
        // Geometric Operations
        // ============================================================================

        /**
         * @brief Project 4D homogeneous coordinates to 3D
         * @return 3D projected coordinates (x/w, y/w, z/w)
         * @note Returns zero vector if w is zero
         */
        half3 project() const noexcept;

        /**
         * @brief Transform to homogeneous coordinates (set w = 1)
         * @return Homogeneous coordinates (x, y, z, 1)
         */
        half4 to_homogeneous() const noexcept;

        // ============================================================================
        // Static Mathematical Functions (SSE Optimized)
        // ============================================================================

        /**
         * @brief Compute dot product of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Dot product result
         */
        static half dot(const half4& a, const half4& b) noexcept;

        /**
         * @brief Compute 3D dot product (ignores w component)
         * @param a First vector
         * @param b Second vector
         * @return 3D dot product result
         */
        static half dot3(const half4& a, const half4& b) noexcept;

        /**
         * @brief Compute 3D cross product (ignores w component)
         * @param a First vector
         * @param b Second vector
         * @return 3D cross product result (w = 0)
         */
        static half4 cross(const half4& a, const half4& b) noexcept;

        /**
         * @brief Linear interpolation between two vectors
         * @param a Start vector
         * @param b End vector
         * @param t Interpolation factor [0, 1]
         * @return Interpolated vector
         */
        static half4 lerp(const half4& a, const half4& b, half t) noexcept;

        /**
         * @brief Linear interpolation between two vectors (float factor)
         * @param a Start vector
         * @param b End vector
         * @param t Interpolation factor [0, 1] as float
         * @return Interpolated vector
         */
        static half4 lerp(const half4& a, const half4& b, float t) noexcept;

        /**
         * @brief Component-wise minimum of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Component-wise minimum
         */
        static half4 min(const half4& a, const half4& b) noexcept;

        /**
         * @brief Component-wise maximum of two vectors
         * @param a First vector
         * @param b Second vector
         * @return Component-wise maximum
         */
        static half4 max(const half4& a, const half4& b) noexcept;

        /**
         * @brief HLSL-like saturate function (clamp components to [0, 1])
         * @param vec Vector to saturate
         * @return Saturated vector
         */
        static half4 saturate(const half4& vec) noexcept;

        // ============================================================================
        // Swizzle Operations (HLSL style)
        // ============================================================================

        /**
         * @brief Swizzle to (x, y)
         * @return 2D vector with x and y components
         */
        half2 xy() const noexcept;

        /**
         * @brief Swizzle to (x, z)
         * @return 2D vector with x and z components
         */
        half2 xz() const noexcept;

        /**
         * @brief Swizzle to (x, w)
         * @return 2D vector with x and w components
         */
        half2 xw() const noexcept;

        /**
         * @brief Swizzle to (y, z)
         * @return 2D vector with y and z components
         */
        half2 yz() const noexcept;

        /**
         * @brief Swizzle to (y, w)
         * @return 2D vector with y and w components
         */
        half2 yw() const noexcept;

        /**
         * @brief Swizzle to (z, w)
         * @return 2D vector with z and w components
         */
        half2 zw() const noexcept;

        /**
         * @brief Swizzle to (x, y, z)
         * @return 3D vector with x, y, z components
         */
        half3 xyz() const noexcept;

        /**
         * @brief Swizzle to (x, y, w)
         * @return 3D vector with x, y, w components
         */
        half3 xyw() const noexcept;

        /**
         * @brief Swizzle to (x, z, w)
         * @return 3D vector with x, z, w components
         */
        half3 xzw() const noexcept;

        /**
         * @brief Swizzle to (y, z, w)
         * @return 3D vector with y, z, w components
         */
        half3 yzw() const noexcept;

        /**
         * @brief Swizzle to (y, x, z, w)
         * @return 4D vector with components rearranged
         */
        half4 yxzw() const noexcept;

        /**
         * @brief Swizzle to (z, x, y, w)
         * @return 4D vector with components rearranged
         */
        half4 zxyw() const noexcept;

        /**
         * @brief Swizzle to (z, y, x, w)
         * @return 4D vector with components rearranged
         */
        half4 zyxw() const noexcept;

        /**
         * @brief Swizzle to (w, z, y, x)
         * @return 4D vector with components rearranged
         */
        half4 wzyx() const noexcept;

        // Color swizzles
        /**
         * @brief Get red component (alias for x)
         * @return Red component
         */
        half r() const noexcept;

        /**
         * @brief Get green component (alias for y)
         * @return Green component
         */
        half g() const noexcept;

        /**
         * @brief Get blue component (alias for z)
         * @return Blue component
         */
        half b() const noexcept;

        /**
         * @brief Get alpha component (alias for w)
         * @return Alpha component
         */
        half a() const noexcept;

        /**
         * @brief Get red and green components
         * @return 2D vector with red and green components
         */
        half2 rg() const noexcept;

        /**
         * @brief Get red and blue components
         * @return 2D vector with red and blue components
         */
        half2 rb() const noexcept;

        /**
         * @brief Get red and alpha components
         * @return 2D vector with red and alpha components
         */
        half2 ra() const noexcept;

        /**
         * @brief Get green and blue components
         * @return 2D vector with green and blue components
         */
        half2 gb() const noexcept;

        /**
         * @brief Get green and alpha components
         * @return 2D vector with green and alpha components
         */
        half2 ga() const noexcept;

        /**
         * @brief Get blue and alpha components
         * @return 2D vector with blue and alpha components
         */
        half2 ba() const noexcept;

        /**
         * @brief Get RGB components
         * @return 3D vector with RGB components
         */
        half3 rgb() const noexcept;

        /**
         * @brief Get red, green, alpha components
         * @return 3D vector with red, green, alpha components
         */
        half3 rga() const noexcept;

        /**
         * @brief Get red, blue, alpha components
         * @return 3D vector with red, blue, alpha components
         */
        half3 rba() const noexcept;

        /**
         * @brief Get green, blue, alpha components
         * @return 3D vector with green, blue, alpha components
         */
        half3 gba() const noexcept;

        /**
         * @brief Get green, red, blue, alpha components
         * @return 4D vector with components rearranged
         */
        half4 grba() const noexcept;

        /**
         * @brief Get blue, red, green, alpha components
         * @return 4D vector with components rearranged
         */
        half4 brga() const noexcept;

        /**
         * @brief Get blue, green, red, alpha components
         * @return 4D vector with components rearranged
         */
        half4 bgra() const noexcept;

        /**
         * @brief Get alpha, blue, green, red components
         * @return 4D vector with components rearranged
         */
        half4 abgr() const noexcept;

        // ============================================================================
        // Utility Methods
        // ============================================================================

        /**
         * @brief Check if vector contains valid finite values
         * @return True if all components are finite (not NaN or infinity)
         */
        bool is_valid() const noexcept;

        /**
         * @brief Check if vector is approximately equal to another
         * @param other Vector to compare with
         * @param epsilon Comparison tolerance
         * @return True if vectors are approximately equal
         */
        bool approximately(const half4& other, float epsilon = Constants::Constants<float>::Epsilon) const noexcept;

        /**
         * @brief Check if vector is approximately zero
         * @param epsilon Comparison tolerance
         * @return True if vector length is approximately zero
         */
        bool approximately_zero(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;

        /**
         * @brief Check if vector is normalized
         * @param epsilon Comparison tolerance
         * @return True if vector length is approximately 1.0
         */
        bool is_normalized(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;

        /**
         * @brief Convert to string representation
         * @return String in format "(x, y, z, w)"
         */
        std::string to_string() const;

        /**
         * @brief Get pointer to raw data
         * @return Pointer to first component
         */
        const half* data() const noexcept;

        /**
         * @brief Get pointer to raw data (mutable)
         * @return Pointer to first component
         */
        half* data() noexcept;

        /**
         * @brief Set x, y, z components from half3
         * @param xyz 3D vector for x, y, z components
         */
        void set_xyz(const half3& xyz) noexcept;

        /**
         * @brief Set x, y components from half2
         * @param xy 2D vector for x, y components
         */
        void set_xy(const half2& xy) noexcept;

        /**
         * @brief Set z, w components from half2
         * @param zw 2D vector for z, w components
         */
        void set_zw(const half2& zw) noexcept;

        // ============================================================================
        // Comparison Operators
        // ============================================================================

        /**
         * @brief Equality comparison
         * @param rhs Vector to compare with
         * @return True if vectors are approximately equal
         */
        bool operator==(const half4& rhs) const noexcept;

        /**
         * @brief Inequality comparison
         * @param rhs Vector to compare with
         * @return True if vectors are not approximately equal
         */
        bool operator!=(const half4& rhs) const noexcept;

        /**
         * @brief Check if any component is infinity
         * @return True if any component is positive or negative infinity
         */
        bool is_inf() const noexcept
        {
            return x.is_inf() || y.is_inf() || z.is_inf() || w.is_inf();
        }

        /**
         * @brief Check if all components are infinity
         * @return True if all components are positive or negative infinity
         */
        bool is_all_inf() const noexcept
        {
            return x.is_inf() && y.is_inf() && z.is_inf() && w.is_inf();
        }

        /**
         * @brief Check if any component is negative infinity
         * @return True if any component is negative infinity
         */
        bool is_negative_inf() const noexcept
        {
            return x.is_negative_inf() || y.is_negative_inf() || z.is_negative_inf() || w.is_negative_inf();
        }

        /**
         * @brief Check if all components are negative infinity
         * @return True if all components are negative infinity
         */
        bool is_all_negative_inf() const noexcept
        {
            return x.is_negative_inf() && y.is_negative_inf() && z.is_negative_inf() && w.is_negative_inf();
        }

        /**
         * @brief Check if any component is positive infinity
         * @return True if any component is positive infinity
         */
        bool is_positive_inf() const noexcept
        {
            return x.is_positive_inf() || y.is_positive_inf() || z.is_positive_inf() || w.is_positive_inf();
        }

        /**
         * @brief Check if all components are positive infinity
         * @return True if all components are positive infinity
         */
        bool is_all_positive_inf() const noexcept
        {
            return x.is_positive_inf() && y.is_positive_inf() && z.is_positive_inf() && w.is_positive_inf();
        }

        /**
         * @brief Check if any component is negative (including negative zero)
         * @return True if any component is negative
         */
        bool is_negative() const noexcept
        {
            return x.is_negative() || y.is_negative() || z.is_negative() || w.is_negative();
        }

        /**
         * @brief Check if all components are negative (including negative zero)
         * @return True if all components are negative
         */
        bool is_all_negative() const noexcept
        {
            return x.is_negative() && y.is_negative() && z.is_negative() && w.is_negative();
        }

        /**
         * @brief Check if any component is positive (excluding negative zero)
         * @return True if any component is positive
         */
        bool is_positive() const noexcept
        {
            return x.is_positive() || y.is_positive() || z.is_positive() || w.is_positive();
        }

        /**
         * @brief Check if all components are positive (excluding negative zero)
         * @return True if all components are positive
         */
        bool is_all_positive() const noexcept
        {
            return x.is_positive() && y.is_positive() && z.is_positive() && w.is_positive();
        }

        /**
         * @brief Check if any component is NaN (Not a Number)
         * @return True if any component is NaN
         */
        bool is_nan() const noexcept
        {
            return x.is_nan() || y.is_nan() || z.is_nan() || w.is_nan();
        }

        /**
         * @brief Check if all components are NaN
         * @return True if all components are NaN
         */
        bool is_all_nan() const noexcept
        {
            return x.is_nan() && y.is_nan() && z.is_nan() && w.is_nan();
        }

        /**
         * @brief Check if any component is finite (not NaN and not infinity)
         * @return True if any component is finite
         */
        bool is_finite() const noexcept 
        {
            return !is_nan() && !is_inf();
        }

        /**
         * @brief Check if all components are finite (not NaN and not infinity)
         * @return True if all components are finite
         */
        bool is_all_finite() const noexcept
        {
            return x.is_finite() && y.is_finite() && z.is_finite() && w.is_finite();
        }

        /**
         * @brief Check if any component is zero (positive or negative)
         * @return True if any component is zero
         */
        bool is_zero() const noexcept
        {
            return x.is_zero() || y.is_zero() || z.is_zero() || w.is_zero();
        }

        /**
         * @brief Check if all components are zero (positive or negative)
         * @return True if all components are zero
         */
        bool is_all_zero() const noexcept
        {
            return x.is_zero() && y.is_zero() && z.is_zero() && w.is_zero();
        }

        /**
         * @brief Check if any component is positive zero
         * @return True if any component is positive zero
         */
        bool is_positive_zero() const noexcept
        {
            return x.is_positive_zero() || y.is_positive_zero() || z.is_positive_zero() || w.is_positive_zero();
        }

        /**
         * @brief Check if all components are positive zero
         * @return True if all components are positive zero
         */
        bool is_all_positive_zero() const noexcept
        {
            return x.is_positive_zero() && y.is_positive_zero() && z.is_positive_zero() && w.is_positive_zero();
        }

        /**
         * @brief Check if any component is negative zero
         * @return True if any component is negative zero
         */
        bool is_negative_zero() const noexcept
        {
            return x.is_negative_zero() || y.is_negative_zero() || z.is_negative_zero() || w.is_negative_zero();
        }

        /**
         * @brief Check if all components are negative zero
         * @return True if all components are negative zero
         */
        bool is_all_negative_zero() const noexcept
        {
            return x.is_negative_zero() && y.is_negative_zero() && z.is_negative_zero() && w.is_negative_zero();
        }

        /**
         * @brief Check if any component is normal (not zero, subnormal, infinity, or NaN)
         * @return True if any component is normal
         */
        bool is_normal() const noexcept 
        {
            if (is_zero() || is_inf() || is_nan()) 
            {
                return false;
            }
            return true;
        }

        /**
         * @brief Check if all components are normal (not zero, subnormal, infinity, or NaN)
         * @return True if all components are normal
         */
        bool is_all_normal() const noexcept
        {
            return x.is_normal() && y.is_normal() && z.is_normal() && w.is_normal();
        }
    };

    // ============================================================================
    // Binary Operators
    // ============================================================================

    /**
     * @brief Vector addition
     * @param lhs Left-hand side vector
     * @param rhs Right-hand side vector
     * @return Result of addition
     */
    inline half4 operator+(half4 lhs, const half4& rhs) noexcept
    {
        return lhs += rhs;
    }

    /**
     * @brief Vector subtraction
     * @param lhs Left-hand side vector
     * @param rhs Right-hand side vector
     * @return Result of subtraction
     */
    inline half4 operator-(half4 lhs, const half4& rhs) noexcept
    {
        return lhs -= rhs;
    }

    /**
     * @brief Component-wise vector multiplication
     * @param lhs Left-hand side vector
     * @param rhs Right-hand side vector
     * @return Result of multiplication
     */
    inline half4 operator*(half4 lhs, const half4& rhs) noexcept
    {
        return lhs *= rhs;
    }

    /**
     * @brief Component-wise vector division
     * @param lhs Left-hand side vector
     * @param rhs Right-hand side vector
     * @return Result of division
     */
    inline half4 operator/(half4 lhs, const half4& rhs) noexcept
    {
        return lhs /= rhs;
    }

    /**
     * @brief Vector-scalar multiplication (half)
     * @param vec Vector to multiply
     * @param scalar Scalar multiplier
     * @return Scaled vector
     */
    inline half4 operator*(half4 vec, half scalar) noexcept
    {
        return vec *= scalar;
    }

    /**
     * @brief Scalar-vector multiplication (half)
     * @param scalar Scalar multiplier
     * @param vec Vector to multiply
     * @return Scaled vector
     */
    inline half4 operator*(half scalar, half4 vec) noexcept
    {
        return vec *= scalar;
    }

    /**
     * @brief Vector-scalar division (half)
     * @param vec Vector to divide
     * @param scalar Scalar divisor
     * @return Scaled vector
     */
    inline half4 operator/(half4 vec, half scalar) noexcept
    {
        return vec /= scalar;
    }

    /**
     * @brief Vector-scalar multiplication (float)
     * @param vec Vector to multiply
     * @param scalar Scalar multiplier
     * @return Scaled vector
     */
    inline half4 operator*(half4 vec, float scalar) noexcept
    {
        return vec *= scalar;
    }

    /**
     * @brief Scalar-vector multiplication (float)
     * @param scalar Scalar multiplier
     * @param vec Vector to multiply
     * @return Scaled vector
     */
    inline half4 operator*(float scalar, half4 vec) noexcept
    {
        return vec *= scalar;
    }

    /**
     * @brief Vector-scalar division (float)
     * @param vec Vector to divide
     * @param scalar Scalar divisor
     * @return Scaled vector
     */
    inline half4 operator/(half4 vec, float scalar) noexcept
    {
        return vec /= scalar;
    }

    // ============================================================================
    // Mixed Type Operators (half4 <-> float4)
    // ============================================================================

    /**
     * @brief Addition between half4 and float4
     * @param lhs half4 vector
     * @param rhs float4 vector
     * @return Result of addition
     */
    inline half4 operator+(const half4& lhs, const float4& rhs) noexcept
    {
        return half4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
    }

    /**
     * @brief Subtraction between half4 and float4
     * @param lhs half4 vector
     * @param rhs float4 vector
     * @return Result of subtraction
     */
    inline half4 operator-(const half4& lhs, const float4& rhs) noexcept
    {
        return half4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
    }

    /**
     * @brief Multiplication between half4 and float4
     * @param lhs half4 vector
     * @param rhs float4 vector
     * @return Result of multiplication
     */
    inline half4 operator*(const half4& lhs, const float4& rhs) noexcept
    {
        return half4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
    }

    /**
     * @brief Division between half4 and float4
     * @param lhs half4 vector
     * @param rhs float4 vector
     * @return Result of division
     */
    inline half4 operator/(const half4& lhs, const float4& rhs) noexcept
    {
        return half4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
    }

    /**
     * @brief Addition between float4 and half4
     * @param lhs float4 vector
     * @param rhs half4 vector
     * @return Result of addition
     */
    inline half4 operator+(const float4& lhs, const half4& rhs) noexcept
    {
        return half4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
    }

    /**
     * @brief Subtraction between float4 and half4
     * @param lhs float4 vector
     * @param rhs half4 vector
     * @return Result of subtraction
     */
    inline half4 operator-(const float4& lhs, const half4& rhs) noexcept
    {
        return half4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
    }

    /**
     * @brief Multiplication between float4 and half4
     * @param lhs float4 vector
     * @param rhs half4 vector
     * @return Result of multiplication
     */
    inline half4 operator*(const float4& lhs, const half4& rhs) noexcept
    {
        return half4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w);
    }

    /**
     * @brief Division between float4 and half4
     * @param lhs float4 vector
     * @param rhs half4 vector
     * @return Result of division
     */
    inline half4 operator/(const float4& lhs, const half4& rhs) noexcept
    {
        return half4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w);
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
    inline half distance(const half4& a, const half4& b) noexcept
    {
        return (b - a).length();
    }

    /**
     * @brief Compute squared distance between two points (faster)
     * @param a First point
     * @param b Second point
     * @return Squared Euclidean distance
     */
    inline half distance_sq(const half4& a, const half4& b) noexcept
    {
        return (b - a).length_sq();
    }

    /**
     * @brief Compute dot product of two vectors
     * @param a First vector
     * @param b Second vector
     * @return Dot product result
     */
    inline half dot(const half4& a, const half4& b) noexcept
    {
        return half4::dot(a, b);
    }

    /**
     * @brief Compute 3D dot product (ignores w component)
     * @param a First vector
     * @param b Second vector
     * @return 3D dot product result
     */
    inline half dot3(const half4& a, const half4& b) noexcept
    {
        return half4::dot3(a, b);
    }

    /**
     * @brief Compute 3D cross product (ignores w component)
     * @param a First vector
     * @param b Second vector
     * @return Cross product result
     */
    inline half4 cross(const half4& a, const half4& b) noexcept
    {
        return half4::cross(a, b);
    }

    /**
     * @brief Normalize vector to unit length
     * @param vec Vector to normalize
     * @return Normalized vector
     */
    inline half4 normalize(const half4& vec) noexcept
    {
        return vec.normalize();
    }

    /**
     * @brief Linear interpolation between two vectors
     * @param a Start vector
     * @param b End vector
     * @param t Interpolation factor [0, 1]
     * @return Interpolated vector
     */
    inline half4 lerp(const half4& a, const half4& b, half t) noexcept
    {
        return half4::lerp(a, b, t);
    }

    /**
     * @brief Linear interpolation between two vectors (float factor)
     * @param a Start vector
     * @param b End vector
     * @param t Interpolation factor [0, 1] as float
     * @return Interpolated vector
     */
    inline half4 lerp(const half4& a, const half4& b, float t) noexcept
    {
        return half4::lerp(a, b, t);
    }

    /**
     * @brief HLSL-like saturate function (clamp components to [0, 1])
     * @param vec Vector to saturate
     * @return Saturated vector
     */
    inline half4 saturate(const half4& vec) noexcept
    {
        return half4::saturate(vec);
    }

    /**
     * @brief Check if two vectors are approximately equal
     * @param a First vector
     * @param b Second vector
     * @param epsilon Comparison tolerance
     * @return True if vectors are approximately equal
     */
    inline bool approximately(const half4& a, const half4& b, float epsilon = Constants::Constants<float>::Epsilon) noexcept
    {
        return a.approximately(b, epsilon);
    }

    /**
     * @brief Check if vector contains valid finite values
     * @param vec Vector to check
     * @return True if vector is valid (finite values)
     */
    inline bool is_valid(const half4& vec) noexcept
    {
        return vec.is_valid();
    }

    /**
     * @brief Check if vector is normalized
     * @param vec Vector to check
     * @param epsilon Comparison tolerance
     * @return True if vector is normalized
     */
    inline bool is_normalized(const half4& vec, float epsilon = Constants::Constants<float>::Epsilon) noexcept
    {
        return vec.is_normalized(epsilon);
    }

    // ============================================================================
    // HLSL-like Global Functions
    // ============================================================================

    /**
     * @brief HLSL-like abs function (component-wise absolute value)
     * @param vec Input vector
     * @return Vector with absolute values of components
     */
    inline half4 abs(const half4& vec) noexcept
    {
        return vec.abs();
    }

    /**
     * @brief HLSL-like sign function (component-wise sign)
     * @param vec Input vector
     * @return Vector with signs of components
     */
    inline half4 sign(const half4& vec) noexcept
    {
        return vec.sign();
    }

    /**
     * @brief HLSL-like floor function (component-wise floor)
     * @param vec Input vector
     * @return Vector with floored components
     */
    inline half4 floor(const half4& vec) noexcept
    {
        return vec.floor();
    }

    /**
     * @brief HLSL-like ceil function (component-wise ceiling)
     * @param vec Input vector
     * @return Vector with ceiling components
     */
    inline half4 ceil(const half4& vec) noexcept
    {
        return vec.ceil();
    }

    /**
     * @brief HLSL-like round function (component-wise rounding)
     * @param vec Input vector
     * @return Vector with rounded components
     */
    inline half4 round(const half4& vec) noexcept
    {
        return vec.round();
    }

    /**
     * @brief HLSL-like frac function (component-wise fractional part)
     * @param vec Input vector
     * @return Vector with fractional parts of components
     */
    inline half4 frac(const half4& vec) noexcept
    {
        return vec.frac();
    }

    /**
     * @brief HLSL-like step function (component-wise step)
     * @param edge Edge value
     * @param vec Input vector
     * @return Step result vector
     */
    inline half4 step(half edge, const half4& vec) noexcept
    {
        return vec.step(edge);
    }

    /**
     * @brief HLSL-like min function (component-wise minimum)
     * @param a First vector
     * @param b Second vector
     * @return Component-wise minimum
     */
    inline half4 min(const half4& a, const half4& b) noexcept
    {
        return half4::min(a, b);
    }

    /**
     * @brief HLSL-like max function (component-wise maximum)
     * @param a First vector
     * @param b Second vector
     * @return Component-wise maximum
     */
    inline half4 max(const half4& a, const half4& b) noexcept
    {
        return half4::max(a, b);
    }

    /**
     * @brief HLSL-like clamp function (component-wise clamping)
     * @param vec Vector to clamp
     * @param min_val Minimum values
     * @param max_val Maximum values
     * @return Clamped vector
     */
    inline half4 clamp(const half4& vec, const half4& min_val, const half4& max_val) noexcept
    {
        return half4::min(half4::max(vec, min_val), max_val);
    }

    /**
     * @brief HLSL-like clamp function (scalar boundaries)
     * @param vec Vector to clamp
     * @param min_val Minimum value
     * @param max_val Maximum value
     * @return Clamped vector
     */
    inline half4 clamp(const half4& vec, float min_val, float max_val) noexcept
    {
        return half4(
            Math::clamp(vec.x, min_val, max_val),
            Math::clamp(vec.y, min_val, max_val),
            Math::clamp(vec.z, min_val, max_val),
            Math::clamp(vec.w, min_val, max_val)
        );
    }

    /**
     * @brief HLSL-like smoothstep function (component-wise smooth interpolation)
     * @param edge0 Lower edge
     * @param edge1 Upper edge
     * @param vec Input vector
     * @return Smoothly interpolated vector
     */
    inline half4 smoothstep(half edge0, half edge1, const half4& vec) noexcept
    {
        return half4(
            Math::smoothstep(edge0, edge1, vec.x),
            Math::smoothstep(edge0, edge1, vec.y),
            Math::smoothstep(edge0, edge1, vec.z),
            Math::smoothstep(edge0, edge1, vec.w)
        );
    }

    // ============================================================================
    // Color Operations
    // ============================================================================

    /**
     * @brief Convert RGB to grayscale using luminance
     * @param rgb RGB color vector
     * @return Grayscale color (RGB = luminance, alpha preserved)
     */
    inline half4 rgb_to_grayscale(const half4& rgb) noexcept
    {
        return rgb.grayscale();
    }

    /**
     * @brief Compute luminance of RGB color
     * @param rgb RGB color vector
     * @return Luminance value
     */
    inline half luminance(const half4& rgb) noexcept
    {
        return rgb.luminance();
    }

    /**
     * @brief Compute average brightness of RGB color
     * @param rgb RGB color vector
     * @return Brightness value
     */
    inline half brightness(const half4& rgb) noexcept
    {
        return rgb.brightness();
    }

    /**
     * @brief Premultiply RGB components by alpha
     * @param color Input color
     * @return Premultiplied color
     */
    inline half4 premultiply_alpha(const half4& color) noexcept
    {
        return color.premultiply_alpha();
    }

    /**
     * @brief Unpremultiply RGB components (divide by alpha)
     * @param color Input color
     * @return Unpremultiplied color
     */
    inline half4 unpremultiply_alpha(const half4& color) noexcept
    {
        return color.unpremultiply_alpha();
    }

    /**
     * @brief Convert sRGB color to linear space
     * @param srgb sRGB color vector
     * @return Linear color values
     */
    inline half4 srgb_to_linear(const half4& srgb) noexcept
    {
        return srgb.srgb_to_linear();
    }

    /**
     * @brief Convert linear color to sRGB space
     * @param linear Linear color vector
     * @return sRGB color values
     */
    inline half4 linear_to_srgb(const half4& linear) noexcept
    {
        return linear.linear_to_srgb();
    }

    // ============================================================================
    // Geometric Operations
    // ============================================================================

    /**
     * @brief Project 4D homogeneous coordinates to 3D
     * @param vec 4D homogeneous vector
     * @return 3D projected coordinates
     */
    inline half3 project(const half4& vec) noexcept
    {
        return vec.project();
    }

    /**
     * @brief Transform to homogeneous coordinates (set w = 1)
     * @param vec 3D vector
     * @return Homogeneous coordinates
     */
    inline half4 to_homogeneous(const half4& vec) noexcept
    {
        return vec.to_homogeneous();
    }

    // ============================================================================
    // Type Conversion Functions
    // ============================================================================

    /**
     * @brief Convert half4 to float4 (promotes components to full precision)
     * @param vec half-precision vector
     * @return full-precision vector
     */
    inline float4 to_float4(const half4& vec) noexcept
    {
        return float4(float(vec.x), float(vec.y), float(vec.z), float(vec.w));
    }

    /**
     * @brief Convert float4 to half4 (demotes components to half precision)
     * @param vec full-precision vector
     * @return half-precision vector
     */
    inline half4 to_half4(const float4& vec) noexcept
    {
        return half4(vec.x, vec.y, vec.z, vec.w);
    }

    // ============================================================================
    // Useful Constants
    // ============================================================================

    /**
     * @brief Zero vector constant (0, 0, 0, 0)
     */
    extern const half4 half4_Zero;

    /**
     * @brief One vector constant (1, 1, 1, 1)
     */
    extern const half4 half4_One;

    /**
     * @brief Unit X vector constant (1, 0, 0, 0)
     */
    extern const half4 half4_UnitX;

    /**
     * @brief Unit Y vector constant (0, 1, 0, 0)
     */
    extern const half4 half4_UnitY;

    /**
     * @brief Unit Z vector constant (0, 0, 1, 0)
     */
    extern const half4 half4_UnitZ;

    /**
     * @brief Unit W vector constant (0, 0, 0, 1)
     */
    extern const half4 half4_UnitW;

    /**
     * @brief Red color constant (1, 0, 0, 1)
     */
    extern const half4 half4_Red;

    /**
     * @brief Green color constant (0, 1, 0, 1)
     */
    extern const half4 half4_Green;

    /**
     * @brief Blue color constant (0, 0, 1, 1)
     */
    extern const half4 half4_Blue;

    /**
     * @brief White color constant (1, 1, 1, 1)
     */
    extern const half4 half4_White;

    /**
     * @brief Black color constant (0, 0, 0, 1)
     */
    extern const half4 half4_Black;

    /**
     * @brief Transparent color constant (0, 0, 0, 0)
     */
    extern const half4 half4_Transparent;

    /**
     * @brief Yellow color constant (1, 1, 0, 1)
     */
    extern const half4 half4_Yellow;

    /**
     * @brief Cyan color constant (0, 1, 1, 1)
     */
    extern const half4 half4_Cyan;

    /**
     * @brief Magenta color constant (1, 0, 1, 1)
     */
    extern const half4 half4_Magenta;

} // namespace Math
