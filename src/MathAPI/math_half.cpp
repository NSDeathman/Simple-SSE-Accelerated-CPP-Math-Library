/**
 * @file math_half.cpp
 * @brief Implementation of 16-bit half-precision floating point type
 * @note Optimized conversion algorithms for performance
 */

#include "math_half.h"

namespace Math {

    // ============================================================================
    // Global Constants Implementation
    // ============================================================================

    const half half_Zero(0.0f);
    const half half_One(1.0f);
    const half half_Max(65504.0f);
    const half half_Min(6.10352e-5f);
    const half half_Epsilon(0.00097656f);
    const half half_PI(Constants::FloatConstants::Pi);
    const half half_TwoPI(Constants::FloatConstants::TwoPi);
    const half half_HalfPI(Constants::FloatConstants::HalfPi);
    const half half_QuarterPI(Constants::FloatConstants::QuarterPi);
    const half half_InvPI(Constants::FloatConstants::InvPi);
    const half half_InvTwoPI(Constants::FloatConstants::InvTwoPi);
    const half half_DegToRad(Constants::FloatConstants::DegToRad);
    const half half_RadToDeg(Constants::FloatConstants::RadToDeg);
    const half half_E(Constants::FloatConstants::E);
    const half half_Sqrt2(Constants::FloatConstants::Sqrt2);
    const half half_Sqrt3(Constants::FloatConstants::Sqrt3);
    const half half_GoldenRatio(Constants::FloatConstants::GoldenRatio);

} // namespace Math
