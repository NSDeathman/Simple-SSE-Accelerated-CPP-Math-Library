#include "math_float3x3.h"
#include "math_float4x4.h"
#include "math_quaternion.h"

namespace Math
{
    // ============================================================================
    // Constructors Implementation
    // ============================================================================

    float3x3::float3x3() noexcept
        : col0_(1.0f, 0.0f, 0.0f, 0.0f),
        col1_(0.0f, 1.0f, 0.0f, 0.0f),
        col2_(0.0f, 0.0f, 1.0f, 0.0f)
    {}

    float3x3::float3x3(const float3& col0, const float3& col1, const float3& col2) noexcept
    {
        col0_ = float4(col0.x, col0.y, col0.z, 0.0f);
        col1_ = float4(col1.x, col1.y, col1.z, 0.0f);
        col2_ = float4(col2.x, col2.y, col2.z, 0.0f);
    }

    float3x3::float3x3(float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22) noexcept
    {
        // m00, m01, m02 = row0
        // m10, m11, m12 = row1  
        // m20, m21, m22 = row2

        // храним в column-major: col0 = [m00, m10, m20], col1 = [m01, m11, m21], etc.
        col0_ = float4(m00, m10, m20, 0.0f);  // col0: m00, m10, m20
        col1_ = float4(m01, m11, m21, 0.0f);  // col1: m01, m11, m21  
        col2_ = float4(m02, m12, m22, 0.0f);  // col2: m02, m12, m22
    }

    float3x3::float3x3(const float* data) noexcept
    {
        col0_ = float4(data[0], data[1], data[2], 0.0f);  // col0: data[0], data[1], data[2]
        col1_ = float4(data[3], data[4], data[5], 0.0f);  // col1: data[3], data[4], data[5]
        col2_ = float4(data[6], data[7], data[8], 0.0f);  // col2: data[6], data[7], data[8]
    }

    float3x3::float3x3(float scalar) noexcept
        : col0_(scalar, 0, 0, 0),
        col1_(0, scalar, 0, 0),
        col2_(0, 0, scalar, 0) {}

    float3x3::float3x3(const float3& diagonal) noexcept
        : col0_(diagonal.x, 0, 0, 0),
        col1_(0, diagonal.y, 0, 0),
        col2_(0, 0, diagonal.z, 0) {}

    float3x3::float3x3(const float4x4& mat4x4) noexcept
    {
        const __m128 mask = _mm_set_ps(0.0f, 1.0f, 1.0f, 1.0f);

        col0_ = float4(_mm_and_ps(mat4x4.col0().get_simd(), mask));
        col1_ = float4(_mm_and_ps(mat4x4.col1().get_simd(), mask));
        col2_ = float4(_mm_and_ps(mat4x4.col2().get_simd(), mask));
    }

    float3x3::float3x3(const quaternion& q) noexcept
    {
        const float xx = q.x * q.x;
        const float yy = q.y * q.y;
        const float zz = q.z * q.z;
        const float xy = q.x * q.y;
        const float xz = q.x * q.z;
        const float yz = q.y * q.z;
        const float wx = q.w * q.x;
        const float wy = q.w * q.y;
        const float wz = q.w * q.z;

        col0_ = float4(1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy), 0.0f);
        col1_ = float4(2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx), 0.0f);
        col2_ = float4(2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy), 0.0f);
    }

    // ============================================================================
    // Assignment Operators Implementation
    // ============================================================================

    float3x3& float3x3::operator=(const float4x4& mat4x4) noexcept
    {
        const __m128 mask = _mm_set_ps(0.0f, 1.0f, 1.0f, 1.0f);
        col0_ = float4(_mm_and_ps(mat4x4.col0().get_simd(), mask));
        col1_ = float4(_mm_and_ps(mat4x4.col1().get_simd(), mask));
        col2_ = float4(_mm_and_ps(mat4x4.col2().get_simd(), mask));
        return *this;
    }

    // ============================================================================
    // Access Operators Implementation
    // ============================================================================

    float3& float3x3::operator[](int colIndex) noexcept
    {
        switch (colIndex) {
        case 0: return *reinterpret_cast<float3*>(&col0_);
        case 1: return *reinterpret_cast<float3*>(&col1_);
        case 2: return *reinterpret_cast<float3*>(&col2_);
        default:
            static float3 dummy;
            return dummy;
        }
    }

    const float3& float3x3::operator[](int colIndex) const noexcept
    {
        switch (colIndex) {
        case 0: return *reinterpret_cast<const float3*>(&col0_);
        case 1: return *reinterpret_cast<const float3*>(&col1_);
        case 2: return *reinterpret_cast<const float3*>(&col2_);
        default:
            static float3 dummy;
            return dummy;
        }
    }

    float& float3x3::operator()(int row, int col) noexcept
    {
        switch (col) {
        case 0:
            switch (row) {
            case 0: return col0_.x;  // (0,0) - row0, col0
            case 1: return col0_.y;  // (1,0) - row1, col0  
            case 2: return col0_.z;  // (2,0) - row2, col0
            }
        case 1:
            switch (row) {
            case 0: return col1_.x;  // (0,1) - row0, col1
            case 1: return col1_.y;  // (1,1) - row1, col1
            case 2: return col1_.z;  // (2,1) - row2, col1
            }
        case 2:
            switch (row) {
            case 0: return col2_.x;  // (0,2) - row0, col2
            case 1: return col2_.y;  // (1,2) - row1, col2
            case 2: return col2_.z;  // (2,2) - row2, col2
            }
        }
        return col0_.x;
    }

    const float& float3x3::operator()(int row, int col) const noexcept
    {
        switch (col) {
        case 0:
            switch (row) {
            case 0: return col0_.x;
            case 1: return col0_.y;
            case 2: return col0_.z;
            }
        case 1:
            switch (row) {
            case 0: return col1_.x;
            case 1: return col1_.y;
            case 2: return col1_.z;
            }
        case 2:
            switch (row) {
            case 0: return col2_.x;
            case 1: return col2_.y;
            case 2: return col2_.z;
            }
        }
        return col0_.x;
    }

    // ============================================================================
    // Column and Row Accessors Implementation
    // ============================================================================

    float3 float3x3::col0() const noexcept { return float3(col0_.x, col0_.y, col0_.z); }
    float3 float3x3::col1() const noexcept { return float3(col1_.x, col1_.y, col1_.z); }
    float3 float3x3::col2() const noexcept { return float3(col2_.x, col2_.y, col2_.z); }

    float3 float3x3::row0() const noexcept { return float3(col0_.x, col1_.x, col2_.x); }
    float3 float3x3::row1() const noexcept { return float3(col0_.y, col1_.y, col2_.y); }
    float3 float3x3::row2() const noexcept { return float3(col0_.z, col1_.z, col2_.z); }

    void float3x3::set_col0(const float3& col) noexcept
    {
        col0_.x = col.x;
        col0_.y = col.y;
        col0_.z = col.z;
    }

    void float3x3::set_col1(const float3& col) noexcept
    {
        col1_.x = col.x;
        col1_.y = col.y;
        col1_.z = col.z;
    }

    void float3x3::set_col2(const float3& col) noexcept
    {
        col2_.x = col.x;
        col2_.y = col.y;
        col2_.z = col.z;
    }

    void float3x3::set_row0(const float3& row) noexcept
    {
        col0_.x = row.x;
        col1_.x = row.y;
        col2_.x = row.z;
    }

    void float3x3::set_row1(const float3& row) noexcept
    {
        col0_.y = row.x;
        col1_.y = row.y;
        col2_.y = row.z;
    }

    void float3x3::set_row2(const float3& row) noexcept
    {
        col0_.z = row.x;
        col1_.z = row.y;
        col2_.z = row.z;
    }

    // ============================================================================
    // Static Constructors Implementation
    // ============================================================================

    float3x3 float3x3::identity() noexcept
    {
        return float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1));
    }

    float3x3 float3x3::zero() noexcept
    {
        return float3x3(float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0));
    }

    float3x3 float3x3::scaling(const float3& scale) noexcept
    {
        return float3x3(float3(scale.x, 0, 0), float3(0, scale.y, 0), float3(0, 0, scale.z));
    }

    float3x3 float3x3::scaling(const float& scaleX, const float& scaleY, const float& scaleZ) noexcept
    {
        return float3x3(float3(scaleX, 0, 0), float3(0, scaleY, 0), float3(0, 0, scaleZ));
    }

    float3x3 float3x3::scaling(float scale) noexcept
    {
        return float3x3(scale);
    }

    float3x3 float3x3::rotation_x(float angle) noexcept
    {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);

        return float3x3(
            float3(1.0f, 0.0f, 0.0f),  // col0
            float3(0.0f, c, s),     // col1
            float3(0.0f, -s, c)      // col2
        );
    }

    float3x3 float3x3::rotation_y(float angle) noexcept
    {
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);

        return float3x3(
            float3(cos_a, 0.0f, -sin_a),  // col0
            float3(0.0f, 1.0f, 0.0f),    // col1
            float3(sin_a, 0.0f, cos_a)    // col2
        );
    }

    float3x3 float3x3::rotation_z(float angle) noexcept
    {
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);

        return float3x3(
            float3(cos_a, sin_a, 0.0f),  // col0
            float3(-sin_a, cos_a, 0.0f),  // col1  
            float3(0.0f, 0.0f, 1.0f)   // col2
        );
    }

    float3x3 float3x3::rotation_axis(const float3& axis, float angle) noexcept
    {
        if (axis.approximately_zero(Constants::Constants<float>::Epsilon)) {
            return identity();
        }

        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);

        const float one_minus_c = 1.0f - c;
        const float3 n = axis.normalize();

        // Предварительные вычисления для минимизации операций
        const float x = n.x, y = n.y, z = n.z;
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float xs = x * s, ys = y * s, zs = z * s;

        // SSE-оптимизированное создание матрицы
        __m128 col0 = _mm_set_ps(0.0f, xz * one_minus_c - ys, xy * one_minus_c + zs, xx * one_minus_c + c);
        __m128 col1 = _mm_set_ps(0.0f, yz * one_minus_c + xs, yy * one_minus_c + c, xy * one_minus_c - zs);
        __m128 col2 = _mm_set_ps(0.0f, zz * one_minus_c + c, yz * one_minus_c - xs, xz * one_minus_c + ys);

        float3x3 result;
        _mm_store_ps(&result.col0_.x, col0);
        _mm_store_ps(&result.col1_.x, col1);
        _mm_store_ps(&result.col2_.x, col2);

        return result;
    }

    float3x3 float3x3::rotation_euler(const float3& angles) noexcept
    {
        return rotation_z(angles.z) * rotation_y(angles.y) * rotation_x(angles.x);
    }

    float3x3 float3x3::skew_symmetric(const float3& vec) noexcept
    {
        return float3x3(
            float3(0, vec.z, -vec.y),  // column 0
            float3(-vec.z, 0, vec.x),  // column 1
            float3(vec.y, -vec.x, 0)   // column 2
        );
    }

    float3x3 float3x3::outer_product(const float3& u, const float3& v) noexcept
    {
        // Внешнее произведение: u * v^T (матрица 3x3)
        // Каждый столбец - это u, умноженный на соответствующий компонент v
        return float3x3(
            float3(u.x * v.x, u.y * v.x, u.z * v.x),  // первый столбец: u * v.x
            float3(u.x * v.y, u.y * v.y, u.z * v.y),  // второй столбец: u * v.y
            float3(u.x * v.z, u.y * v.z, u.z * v.z)   // третий столбец: u * v.z
        );
    }

    // ============================================================================
    // Compound Assignment Operators Implementation
    // ============================================================================

    float3x3& float3x3::operator+=(const float3x3& rhs) noexcept
    {
        col0_ += rhs.col0_;
        col1_ += rhs.col1_;
        col2_ += rhs.col2_;
        return *this;
    }

    float3x3& float3x3::operator-=(const float3x3& rhs) noexcept
    {
        col0_ -= rhs.col0_;
        col1_ -= rhs.col1_;
        col2_ -= rhs.col2_;
        return *this;
    }

    float3x3& float3x3::operator*=(float scalar) noexcept
    {
        col0_ *= scalar;
        col1_ *= scalar;
        col2_ *= scalar;
        return *this;
    }

    float3x3& float3x3::operator/=(float scalar) noexcept
    {
        const float inv_scalar = 1.0f / scalar;
        col0_ *= inv_scalar;
        col1_ *= inv_scalar;
        col2_ *= inv_scalar;
        return *this;
    }

    float3x3& float3x3::operator*=(const float3x3& rhs) noexcept
    {
        *this = *this * rhs;
        return *this;
    }

    // ============================================================================
    // Unary Operators Implementation
    // ============================================================================

    float3x3 float3x3::operator+() const noexcept { return *this; }

    float3x3 float3x3::operator-() const noexcept
    {
        return float3x3(
            -col0_.xyz(),
            -col1_.xyz(),
            -col2_.xyz()
        );
    }

    // ============================================================================
    // Matrix Operations Implementation
    // ============================================================================

    float3x3 float3x3::transposed() const noexcept
    {
        return float3x3(
            row0(),  // первая строка становится первым столбцом
            row1(),  // вторая строка становится вторым столбцом  
            row2()   // третья строка становится третьим столбцом
        );
    }

    float float3x3::determinant() const noexcept
    {
        const float3 col0 = col0_.xyz();
        const float3 col1 = col1_.xyz();
        const float3 col2 = col2_.xyz();

        const float a = col0.x, b = col0.y, c = col0.z;
        const float d = col1.x, e = col1.y, f = col1.z;
        const float g = col2.x, h = col2.y, i = col2.z;

        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    }

    float3x3 float3x3::inverted() const noexcept
    {
        // Загружаем столбцы матрицы
        __m128 c0 = _mm_load_ps(&col0_.x);
        __m128 c1 = _mm_load_ps(&col1_.x);
        __m128 c2 = _mm_load_ps(&col2_.x);

        // Вычисляем определитель по формуле Саррюса через SSE
        // det = a(ei − fh) − b(di − fg) + c(dh − eg)

        // Переставляем элементы для вычисления миноров 2x2
        __m128 c1_yzx = _mm_shuffle_ps(c1, c1, _MM_SHUFFLE(3, 0, 2, 1)); // [e, f, d, w]
        __m128 c2_yzx = _mm_shuffle_ps(c2, c2, _MM_SHUFFLE(3, 0, 2, 1)); // [h, i, g, w]
        __m128 c1_zxy = _mm_shuffle_ps(c1, c1, _MM_SHUFFLE(3, 1, 0, 2)); // [f, d, e, w]
        __m128 c2_zxy = _mm_shuffle_ps(c2, c2, _MM_SHUFFLE(3, 1, 0, 2)); // [i, g, h, w]

        // Вычисляем произведения для миноров
        __m128 minor_products = _mm_sub_ps(_mm_mul_ps(c1_yzx, c2_zxy),
            _mm_mul_ps(c1_zxy, c2_yzx));

        // Умножаем на элементы первого столбца и знаки
        __m128 det_terms = _mm_mul_ps(c0, minor_products);
        __m128 signed_terms = _mm_mul_ps(det_terms, _mm_set_ps(0.0f, 1.0f, -1.0f, 1.0f));

        // Горизонтальное суммирование
        __m128 det_sum = _mm_hadd_ps(signed_terms, signed_terms);
        det_sum = _mm_hadd_ps(det_sum, det_sum);
        float det = _mm_cvtss_f32(det_sum);

        if (std::abs(det) < Constants::Constants<float>::Epsilon) {
            return identity();
        }

        const float inv_det = 1.0f / det;
        const __m128 inv_det_vec = _mm_set1_ps(inv_det);

        // Вычисляем матрицу алгебраических дополнений (adjugate matrix)
        // Формула для обратной матрицы 3x3: A⁻¹ = (1/det) * adj(A)

        // Алгебраические дополнения для каждого элемента:
        // adj00 = (e*i - f*h)
        // adj01 = -(b*i - c*h) 
        // adj02 = (b*f - c*e)
        // adj10 = -(d*i - f*g)
        // adj11 = (a*i - c*g)
        // adj12 = -(a*f - c*d)
        // adj20 = (d*h - e*g)
        // adj21 = -(a*h - b*g)
        // adj22 = (a*e - b*d)

        // Извлекаем компоненты для удобства
        const float a = col0_.x, b = col0_.y, c = col0_.z;
        const float d = col1_.x, e = col1_.y, f = col1_.z;
        const float g = col2_.x, h = col2_.y, i = col2_.z;

        // Вычисляем все алгебраические дополнения одновременно через SSE
        __m128 adj00_adj01 = _mm_set_ps(0.0f,
            b * f - c * e,    // adj02
            -(b * i - c * h),  // adj01  
            e * i - f * h);   // adj00

        __m128 adj10_adj11 = _mm_set_ps(0.0f,
            -(a * f - c * d),  // adj12
            a * i - c * g,    // adj11
            -(d * i - f * g)); // adj10

        __m128 adj20_adj21 = _mm_set_ps(0.0f,
            a * e - b * d,    // adj22
            -(a * h - b * g),  // adj21
            d * h - e * g);   // adj20

        // Умножаем на обратный определитель
        adj00_adj01 = _mm_mul_ps(adj00_adj01, inv_det_vec);
        adj10_adj11 = _mm_mul_ps(adj10_adj11, inv_det_vec);
        adj20_adj21 = _mm_mul_ps(adj20_adj21, inv_det_vec);

        // Транспонируем матрицу (adjugate уже транспонирована по отношению к cofactor matrix)
        // Столбец 0: [adj00, adj10, adj20]
        __m128 result_col0 = _mm_set_ps(0.0f,
            _mm_cvtss_f32(_mm_shuffle_ps(adj20_adj21, adj20_adj21, _MM_SHUFFLE(0, 0, 0, 0))), // adj20
            _mm_cvtss_f32(_mm_shuffle_ps(adj10_adj11, adj10_adj11, _MM_SHUFFLE(0, 0, 0, 0))), // adj10
            _mm_cvtss_f32(adj00_adj01)); // adj00

        // Столбец 1: [adj01, adj11, adj21]
        __m128 result_col1 = _mm_set_ps(0.0f,
            _mm_cvtss_f32(_mm_shuffle_ps(adj20_adj21, adj20_adj21, _MM_SHUFFLE(1, 1, 1, 1))), // adj21
            _mm_cvtss_f32(_mm_shuffle_ps(adj10_adj11, adj10_adj11, _MM_SHUFFLE(1, 1, 1, 1))), // adj11
            _mm_cvtss_f32(_mm_shuffle_ps(adj00_adj01, adj00_adj01, _MM_SHUFFLE(1, 1, 1, 1)))); // adj01

        // Столбец 2: [adj02, adj12, adj22]
        __m128 result_col2 = _mm_set_ps(0.0f,
            _mm_cvtss_f32(_mm_shuffle_ps(adj20_adj21, adj20_adj21, _MM_SHUFFLE(2, 2, 2, 2))), // adj22
            _mm_cvtss_f32(_mm_shuffle_ps(adj10_adj11, adj10_adj11, _MM_SHUFFLE(2, 2, 2, 2))), // adj12
            _mm_cvtss_f32(_mm_shuffle_ps(adj00_adj01, adj00_adj01, _MM_SHUFFLE(2, 2, 2, 2)))); // adj02

        float3x3 result;
        _mm_store_ps(&result.col0_.x, result_col0);
        _mm_store_ps(&result.col1_.x, result_col1);
        _mm_store_ps(&result.col2_.x, result_col2);

        return result;
    }

    float3x3 float3x3::normal_matrix(const float3x3& model) noexcept
    {
        float3x3 inv = model.inverted();
        float3x3 result = inv.transposed();

        float3 col0 = result.col0().normalize();
        float3 col1 = result.col1().normalize();
        float3 col2 = result.col2().normalize();

        return float3x3(col0, col1, col2);
    }

    float float3x3::trace() const noexcept
    {
        return col0_.x + col1_.y + col2_.z;
    }

    float3 float3x3::diagonal() const noexcept
    {
        return float3(col0_.x, col1_.y, col2_.z);
    }

    float float3x3::frobenius_norm() const noexcept
    {
        return std::sqrt(col0_.length_sq() + col1_.length_sq() + col2_.length_sq());
    }

    float3x3 float3x3::symmetric_part() const noexcept
    {
        float3x3 trans = transposed();
        return (*this + trans) * 0.5f;
    }

    float3x3 float3x3::skew_symmetric_part() const noexcept
    {
        float3x3 trans = transposed();
        return (*this - trans) * 0.5f;
    }

    // ============================================================================
    // Vector Transformations Implementation
    // ============================================================================

    float3 float3x3::transform_vector(const float3& vec) const noexcept
    {
        // Загружаем вектор и broadcast компоненты
        __m128 v = _mm_set_ps(0.0f, vec.z, vec.y, vec.x);
        __m128 vx = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 vy = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 vz = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

        // Загружаем столбцы матрицы
        __m128 col0 = _mm_load_ps(&col0_.x);
        __m128 col1 = _mm_load_ps(&col1_.x);
        __m128 col2 = _mm_load_ps(&col2_.x);

        // Умножаем и складываем
        __m128 result = _mm_mul_ps(col0, vx);
        result = _mm_add_ps(result, _mm_mul_ps(col1, vy));
        result = _mm_add_ps(result, _mm_mul_ps(col2, vz));

        return float3(result);
    }

    float3 float3x3::transform_point(const float3& point) const noexcept
    {
        return transform_vector(point);
    }

    float3 float3x3::transform_normal(const float3& normal) const noexcept
    {
        __m128 n = _mm_set_ps(0.0f, normal.z, normal.y, normal.x);

        // Загружаем строки (транспонированная матрица)
        __m128 row0 = _mm_set_ps(0.0f, col2_.x, col1_.x, col0_.x);
        __m128 row1 = _mm_set_ps(0.0f, col2_.y, col1_.y, col0_.y);
        __m128 row2 = _mm_set_ps(0.0f, col2_.z, col1_.z, col0_.z);

        // Умножаем и суммируем
        __m128 result = _mm_mul_ps(row0, _mm_shuffle_ps(n, n, _MM_SHUFFLE(0, 0, 0, 0)));
        result = _mm_add_ps(result, _mm_mul_ps(row1, _mm_shuffle_ps(n, n, _MM_SHUFFLE(1, 1, 1, 1))));
        result = _mm_add_ps(result, _mm_mul_ps(row2, _mm_shuffle_ps(n, n, _MM_SHUFFLE(2, 2, 2, 2))));

        // Нормализуем результат
        __m128 len = _mm_sqrt_ps(_mm_dp_ps(result, result, 0x7F));
        result = _mm_div_ps(result, len);

        return float3(result);
    }

    // ============================================================================
    // Decomposition Methods Implementation
    // ============================================================================

    float3 float3x3::extract_scale() const noexcept
    {
        return float3(col0_.length(), col1_.length(), col2_.length());
    }

    float3x3 float3x3::extract_rotation() const noexcept
    {
        // Быстрая проверка: если матрица уже ортонормирована, возвращаем как есть
        if (is_orthonormal(1e-4f)) {
            return *this;
        }

        // Загружаем столбцы
        __m128 c0 = _mm_load_ps(&col0_.x);
        __m128 c1 = _mm_load_ps(&col1_.x);
        __m128 c2 = _mm_load_ps(&col2_.x);

        // Нормализуем первый столбец с помощью _mm_dp_ps
        __m128 len0 = _mm_sqrt_ps(_mm_dp_ps(c0, c0, 0x7F));
        c0 = _mm_div_ps(c0, len0);

        // Ортогонализируем второй столбец
        __m128 dot01 = _mm_dp_ps(c1, c0, 0x7F);
        c1 = _mm_sub_ps(c1, _mm_mul_ps(dot01, c0));
        __m128 len1 = _mm_sqrt_ps(_mm_dp_ps(c1, c1, 0x7F));
        c1 = _mm_div_ps(c1, len1);

        // Третий столбец через векторное произведение (гарантированно ортогонален)
        // cross(c0, c1) = [c0.y*c1.z - c0.z*c1.y, c0.z*c1.x - c0.x*c1.z, c0.x*c1.y - c0.y*c1.x]
        __m128 c0_yzx = _mm_shuffle_ps(c0, c0, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c1_yzx = _mm_shuffle_ps(c1, c1, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c0_zxy = _mm_shuffle_ps(c0, c0, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 c1_zxy = _mm_shuffle_ps(c1, c1, _MM_SHUFFLE(3, 1, 0, 2));

        c2 = _mm_sub_ps(_mm_mul_ps(c0_yzx, c1_zxy), _mm_mul_ps(c0_zxy, c1_yzx));

        float3x3 result;
        _mm_store_ps(&result.col0_.x, c0);
        _mm_store_ps(&result.col1_.x, c1);
        _mm_store_ps(&result.col2_.x, c2);

        return result;
    }

    // ============================================================================
    // Utility Methods Implementation
    // ============================================================================

    bool float3x3::is_identity(float epsilon) const noexcept
    {
        return col0_.approximately(float4(1, 0, 0, 0), epsilon) &&
            col1_.approximately(float4(0, 1, 0, 0), epsilon) &&
            col2_.approximately(float4(0, 0, 1, 0), epsilon);
    }

    bool float3x3::is_orthogonal(float epsilon) const noexcept
    {
        return MathFunctions::approximately(dot(col0(), col1()), 0.0f, epsilon) &&
            MathFunctions::approximately(dot(col0(), col2()), 0.0f, epsilon) &&
            MathFunctions::approximately(dot(col1(), col2()), 0.0f, epsilon);
    }

    bool float3x3::is_orthonormal(float epsilon) const noexcept
    {
        return is_orthogonal(epsilon) &&
            MathFunctions::approximately(col0().length_sq(), 1.0f, epsilon) &&
            MathFunctions::approximately(col1().length_sq(), 1.0f, epsilon) &&
            MathFunctions::approximately(col2().length_sq(), 1.0f, epsilon);
    }

    bool float3x3::approximately(const float3x3& other, float epsilon) const noexcept
    {
        return
            MathFunctions::approximately(col0_.x, other.col0_.x, epsilon) &&
            MathFunctions::approximately(col0_.y, other.col0_.y, epsilon) &&
            MathFunctions::approximately(col0_.z, other.col0_.z, epsilon) &&
            MathFunctions::approximately(col1_.x, other.col1_.x, epsilon) &&
            MathFunctions::approximately(col1_.y, other.col1_.y, epsilon) &&
            MathFunctions::approximately(col1_.z, other.col1_.z, epsilon) &&
            MathFunctions::approximately(col2_.x, other.col2_.x, epsilon) &&
            MathFunctions::approximately(col2_.y, other.col2_.y, epsilon) &&
            MathFunctions::approximately(col2_.z, other.col2_.z, epsilon);
    }

    std::string float3x3::to_string() const
    {
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer),
            "[%8.4f, %8.4f, %8.4f]\n"   // row0: (0,0), (0,1), (0,2)
            "[%8.4f, %8.4f, %8.4f]\n"   // row1: (1,0), (1,1), (1,2)
            "[%8.4f, %8.4f, %8.4f]",    // row2: (2,0), (2,1), (2,2)
            (*this)(0, 0), (*this)(0, 1), (*this)(0, 2),  // row0
            (*this)(1, 0), (*this)(1, 1), (*this)(1, 2),  // row1  
            (*this)(2, 0), (*this)(2, 1), (*this)(2, 2)); // row2
        return std::string(buffer);
    }

    void float3x3::to_column_major(float* data) const noexcept
    {
        data[0] = col0_.x; data[1] = col0_.y; data[2] = col0_.z;
        data[3] = col1_.x; data[4] = col1_.y; data[5] = col1_.z;
        data[6] = col2_.x; data[7] = col2_.y; data[8] = col2_.z;
    }

    void float3x3::to_row_major(float* data) const noexcept
    {
        data[0] = col0_.x; data[1] = col1_.x; data[2] = col2_.x;
        data[3] = col0_.y; data[4] = col1_.y; data[5] = col2_.y;
        data[6] = col0_.z; data[7] = col1_.z; data[8] = col2_.z;
    }

    // ============================================================================
    // Comparison Operators Implementation
    // ============================================================================

    bool float3x3::operator==(const float3x3& rhs) const noexcept
    {
        return approximately(rhs);
    }

    bool float3x3::operator!=(const float3x3& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    // ============================================================================
    // Global Constants Implementation
    // ============================================================================

    const float3x3 float3x3_Identity = float3x3::identity();
    const float3x3 float3x3_Zero = float3x3::zero();
} // namespace Math
