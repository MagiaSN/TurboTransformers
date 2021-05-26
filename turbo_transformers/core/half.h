// Copyright (C) 2020 THL A29 Limited, a Tencent company.
// All rights reserved.
// Licensed under the BSD 3-Clause License (the "License"); you may
// not use this file except in compliance with the License. You may
// obtain a copy of the License at
// https://opensource.org/licenses/BSD-3-Clause
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" basis,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// See the AUTHORS file for names of contributors.

#pragma once

#include <fp16.h>

#if defined(__CUDA_ARCH__)
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#include <cuda_fp16.h>
#else
#define HOSTDEVICE
#define DEVICE
#endif

namespace turbo_transformers {
namespace core {

struct alignas(2) Half {
  std::uint16_t x;

  Half() = default;

  Half(const Half &f) = default;

  Half &operator=(const Half &f) = default;

  Half &operator=(Half &&f) = default;

  ~Half() = default;

  explicit HOSTDEVICE Half(float other) {
#if defined(__CUDA_ARCH__)
    half tmp = __float2half(other);
    x = *reinterpret_cast<uint16_t *>(&tmp);
#else
    x = fp16_ieee_from_fp32_value(other);
#endif
  }

#if defined(__CUDA_ARCH__)
  explicit DEVICE Half(__half other) {
    x = *reinterpret_cast<uint16_t *>(&other);
  }
#endif

  inline HOSTDEVICE Half &operator+=(const Half &other) {
#if defined(__CUDA_ARCH__)
    const __half a = *reinterpret_cast<const __half *>(&x);
    const __half b = *reinterpret_cast<const __half *>(&other.x);
    __half sum = __hadd(a, b);
    x = *reinterpret_cast<uint16_t *>(&sum);
#else
    // Basically fp16 should only be used on GPU...
    const float a = fp16_ieee_to_fp32_value(x);
    const float b = fp16_ieee_to_fp32_value(other.x);
    float sum = a + b;
    x = fp16_ieee_from_fp32_value(sum);
#endif
    return *this;
  }

  explicit inline HOSTDEVICE operator float() const {
#if defined(__CUDA_ARCH__)
    return __half2float(x);
#else
    return fp16_ieee_to_fp32_value(x);
#endif
  }
};  // struct Half

inline HOSTDEVICE Half operator+(const Half &a, const Half &b) {
#if defined(__CUDA_ARCH__)
  const __half half_a = *reinterpret_cast<const __half *>(&a.x);
  const __half half_b = *reinterpret_cast<const __half *>(&b.x);
  __half sum = __hadd(half_a, half_b);
#else
  const float fp32_a = fp16_ieee_to_fp32_value(a.x);
  const float fp32_b = fp16_ieee_to_fp32_value(b.x);
  float sum = fp32_a + fp32_b;
#endif
  return Half(sum);
}

inline HOSTDEVICE Half operator-(const Half &a, const Half &b) {
#if defined(__CUDA_ARCH__)
  const __half half_a = *reinterpret_cast<const __half *>(&a.x);
  const __half half_b = *reinterpret_cast<const __half *>(&b.x);
  __half sub = __hsub(half_a, half_b);
#else
  const float fp32_a = fp16_ieee_to_fp32_value(a.x);
  const float fp32_b = fp16_ieee_to_fp32_value(b.x);
  float sub = fp32_a - fp32_b;
#endif
  return Half(sub);
}

inline HOSTDEVICE Half operator*(const Half &a, const Half &b) {
#if defined(__CUDA_ARCH__)
  const __half half_a = *reinterpret_cast<const __half *>(&a.x);
  const __half half_b = *reinterpret_cast<const __half *>(&b.x);
  __half mul = __hmul(half_a, half_b);
#else
  const float fp32_a = fp16_ieee_to_fp32_value(a.x);
  const float fp32_b = fp16_ieee_to_fp32_value(b.x);
  float mul = fp32_a * fp32_b;
#endif
  return Half(mul);
}

inline HOSTDEVICE Half operator/(const Half &a, const Half &b) {
#if defined(__CUDA_ARCH__)
  const __half half_a = *reinterpret_cast<const __half *>(&a.x);
  const __half half_b = *reinterpret_cast<const __half *>(&b.x);
  __half div = __hdiv(half_a, half_b);
#else
  const float fp32_a = fp16_ieee_to_fp32_value(a.x);
  const float fp32_b = fp16_ieee_to_fp32_value(b.x);
  float div = fp32_a / fp32_b;
#endif
  return Half(div);
}

}  // namespace core
}  // namespace turbo_transformers
