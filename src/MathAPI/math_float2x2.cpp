#include "math_float2x2.h"
#include "math_float2.h"

namespace Math
{
    // ============================================================================
    // Constructors
    // ============================================================================

    float2x2::float2x2() noexcept
        : col0_(1.0f, 0.0f), col1_(0.0f, 1.0f) {}

    float2x2::float2x2(const float2& col0, const float2& col1) noexcept
        : col0_(col0), col1_(col1) {}

    float2x2::float2x2(float col0x, float col0y, float col1x, float col1y) noexcept
        : col0_(col0x, col0y), col1_(col1x, col1y) {}

    float2x2::float2x2(const float* data) noexcept
        : col0_(data[0], data[1]), col1_(data[2], data[3]) {}

    float2x2::float2x2(float scalar) noexcept
        : col0_(scalar, 0.0f), col1_(0.0f, scalar) {}

    float2x2::float2x2(const float2& diagonal) noexcept
        : col0_(diagonal.x, 0.0f), col1_(0.0f, diagonal.y) {}

    float2x2::float2x2(__m128 sse_data) noexcept
    {
        set_sse_data(sse_data);
    }

    // ============================================================================
    // Static Constructors
    // ============================================================================

    float2x2 float2x2::identity() noexcept
    {
        return float2x2(float2(1.0f, 0.0f), float2(0.0f, 1.0f));
    }

    float2x2 float2x2::zero() noexcept
    {
        return float2x2(float2(0.0f, 0.0f), float2(0.0f, 0.0f));
    }

    float2x2 float2x2::rotation(float angle) noexcept
    {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);
        return float2x2(float2(c, s), float2(-s, c));
    }

    float2x2 float2x2::scaling(const float2& scale) noexcept
    {
        return float2x2(float2(scale.x, 0.0f), float2(0.0f, scale.y));
    }

    float2x2 float2x2::scaling(float x, float y) noexcept
    {
        return scaling(float2(x, y));
    }

    float2x2 float2x2::scaling(float uniformScale) noexcept
    {
        return scaling(float2(uniformScale, uniformScale));
    }

    float2x2 float2x2::shear(const float2& shear) noexcept
    {
        return float2x2(float2(1.0f, shear.y), float2(shear.x, 1.0f));
    }

    float2x2 float2x2::shear(float x, float y) noexcept
    {
        return shear(float2(x, y));
    }

    // ============================================================================
    // Access Operators
    // ============================================================================

    float2& float2x2::operator[](int colIndex) noexcept
    {
        return (colIndex == 0) ? col0_ : col1_;
    }

    const float2& float2x2::operator[](int colIndex) const noexcept
    {
        return (colIndex == 0) ? col0_ : col1_;
    }

    float& float2x2::operator()(int row, int col) noexcept
    {
        return (col == 0) ?
            (row == 0 ? col0_.x : col0_.y) :
            (row == 0 ? col1_.x : col1_.y);
    }

    const float& float2x2::operator()(int row, int col) const noexcept
    {
        return (col == 0) ?
            (row == 0 ? col0_.x : col0_.y) :
            (row == 0 ? col1_.x : col1_.y);
    }

    // ============================================================================
    // Column and Row Accessors
    // ============================================================================

    float2 float2x2::col0() const noexcept { return col0_; }
    float2 float2x2::col1() const noexcept { return col1_; }

    float2 float2x2::row0() const noexcept { return float2(col0_.x, col1_.x); }
    float2 float2x2::row1() const noexcept { return float2(col0_.y, col1_.y); }

    void float2x2::set_col0(const float2& col) noexcept { col0_ = col; }
    void float2x2::set_col1(const float2& col) noexcept { col1_ = col; }

    void float2x2::set_row0(const float2& row) noexcept
    {
        col0_.x = row.x;
        col1_.x = row.y;
    }

    void float2x2::set_row1(const float2& row) noexcept
    {
        col0_.y = row.x;
        col1_.y = row.y;
    }

    // ============================================================================
    // SSE Accessors
    // ============================================================================

    __m128 float2x2::sse_data() const noexcept
    {
        return _mm_setr_ps(col0_.x, col0_.y, col1_.x, col1_.y);
    }

    void float2x2::set_sse_data(__m128 sse_data) noexcept
    {
        float temp[4];
        _mm_store_ps(temp, sse_data);
        col0_.x = temp[0]; col0_.y = temp[1];
        col1_.x = temp[2]; col1_.y = temp[3];
    }

    // ============================================================================
    // Compound Assignment Operators
    // ============================================================================

    float2x2& float2x2::operator+=(const float2x2& rhs) noexcept
    {
        __m128 result = _mm_add_ps(this->sse_data(), rhs.sse_data());
        this->set_sse_data(result);
        return *this;
    }

    float2x2& float2x2::operator-=(const float2x2& rhs) noexcept
    {
        __m128 result = _mm_sub_ps(this->sse_data(), rhs.sse_data());
        this->set_sse_data(result);
        return *this;
    }

    float2x2& float2x2::operator*=(float scalar) noexcept
    {
        col0_ *= scalar;
        col1_ *= scalar;
        return *this;
    }

    float2x2& float2x2::operator/=(float scalar) noexcept
    {
        const float inv_scalar = 1.0f / scalar;
        col0_ *= inv_scalar;
        col1_ *= inv_scalar;
        return *this;
    }

    float2x2& float2x2::operator*=(const float2x2& rhs) noexcept
    {
        *this = *this * rhs;
        return *this;
    }

    // ============================================================================
    // Unary Operators
    // ============================================================================

    float2x2 float2x2::operator+() const noexcept { return *this; }

    float2x2 float2x2::operator-() const noexcept
    {
        return float2x2(-col0_, -col1_);
    }

    // ============================================================================
    // Matrix Operations
    // ============================================================================

    float2x2 float2x2::transposed() const noexcept
    {
        return float2x2(
            float2(col0_.x, col1_.x),
            float2(col0_.y, col1_.y)
        );
    }

    float float2x2::determinant() const noexcept
    {
        return col0_.x * col1_.y - col0_.y * col1_.x;
    }

    float2x2 float2x2::inverted() const noexcept
    {
        const float det = determinant();
        if (std::abs(det) < Constants::Constants<float>::Epsilon)
        {
            return identity();
        }

        const float inv_det = 1.0f / det;
        return float2x2(
            float2(col1_.y, -col0_.y) * inv_det,
            float2(-col1_.x, col0_.x) * inv_det
        );
    }

    float2x2 float2x2::adjugate() const noexcept
    {
        return float2x2(
            float2(col1_.y, -col0_.y),
            float2(-col1_.x, col0_.x)
        );
    }

    float float2x2::trace() const noexcept
    {
        return col0_.x + col1_.y;
    }

    float2 float2x2::diagonal() const noexcept
    {
        return float2(col0_.x, col1_.y);
    }

    float float2x2::frobenius_norm() const noexcept
    {
        return std::sqrt(col0_.length_sq() + col1_.length_sq());
    }

    // ============================================================================
    // Vector Transformations
    // ============================================================================

    float2 float2x2::transform_vector(const float2& vec) const noexcept
    {
        return float2(
            col0_.x * vec.x + col1_.x * vec.y,
            col0_.y * vec.x + col1_.y * vec.y
        );
    }

    float2 float2x2::transform_point(const float2& point) const noexcept
    {
        return transform_vector(point);
    }

    // ============================================================================
    // Transformation Component Extraction
    // ============================================================================

    float float2x2::get_rotation() const noexcept
    {
        float2 x_axis = col0_.normalize();
        return std::atan2(x_axis.y, x_axis.x);
    }

    float2 float2x2::get_scale() const noexcept
    {
        return float2(col0_.length(), col1_.length());
    }

    void float2x2::set_rotation(float angle) noexcept
    {
        float2 current_scale = get_scale();
        float cos_angle = std::cos(angle);
        float sin_angle = std::sin(angle);

        col0_ = float2(cos_angle, sin_angle) * current_scale.x;
        col1_ = float2(-sin_angle, cos_angle) * current_scale.y;
    }

    void float2x2::set_scale(const float2& scale) noexcept
    {
        float2 current_scale = get_scale();
        if (current_scale.x > 0) col0_ = col0_ * (scale.x / current_scale.x);
        if (current_scale.y > 0) col1_ = col1_ * (scale.y / current_scale.y);
    }

    // ============================================================================
    // Utility Methods
    // ============================================================================

    bool float2x2::is_identity(float epsilon) const noexcept
    {
        return col0_.approximately(float2(1.0f, 0.0f), epsilon) &&
            col1_.approximately(float2(0.0f, 1.0f), epsilon);
    }

    bool float2x2::is_orthogonal(float epsilon) const noexcept
    {
        return MathFunctions::approximately(dot(col0_, col1_), 0.0f, epsilon);
    }

    bool float2x2::is_rotation(float epsilon) const noexcept
    {
        return is_orthogonal(epsilon) &&
            MathFunctions::approximately(col0_.length_sq(), 1.0f, epsilon) &&
            MathFunctions::approximately(col1_.length_sq(), 1.0f, epsilon) &&
            MathFunctions::approximately(determinant(), 1.0f, epsilon);
    }

    bool float2x2::approximately(const float2x2& other, float epsilon) const noexcept
    {
        return col0_.approximately(other.col0_, epsilon) &&
            col1_.approximately(other.col1_, epsilon);
    }

    bool float2x2::approximately_zero(float epsilon) const noexcept
    {
        return col0_.approximately_zero(epsilon) &&
            col1_.approximately_zero(epsilon);
    }

    std::string float2x2::to_string() const
    {
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer),
            "[%8.4f, %8.4f]\n"
            "[%8.4f, %8.4f]",
            col0_.x, col1_.x,
            col0_.y, col1_.y);
        return std::string(buffer);
    }

    void float2x2::to_column_major(float* data) const noexcept
    {
        data[0] = col0_.x;
        data[1] = col0_.y;
        data[2] = col1_.x;
        data[3] = col1_.y;
    }

    void float2x2::to_row_major(float* data) const noexcept
    {
        data[0] = col0_.x;
        data[1] = col1_.x;
        data[2] = col0_.y;
        data[3] = col1_.y;
    }

    // ============================================================================
    // Comparison Operators
    // ============================================================================

    bool float2x2::operator==(const float2x2& rhs) const noexcept
    {
        return approximately(rhs);
    }

    bool float2x2::operator!=(const float2x2& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    // ============================================================================
    // Useful Constants
    // ============================================================================

    const float2x2 float2x2_Identity = float2x2::identity();
    const float2x2 float2x2_Zero = float2x2::zero();
} // namespace Math
