#pragma once
#include <cmath>
#include <cstdlib>
#include <new>
#include "ops.h"

// unary
template<typename T>
T ops<T>::square(T a) { return a * a; }
template<typename T>
T ops<T>::log(T a) { return std::log(a); }
template<typename T>
T ops<T>::exp(T a) { return std::exp(a); }
template<typename T>
T ops<T>::sqrt(T a) { return std::sqrt(a); }
// binary
template<typename T>
T ops<T>::add(T a, T b) { return a + b; }
template<typename T>
T ops<T>::sub(T a, T b) { return a - b; }
template<typename T>
T ops<T>::mul(T a, T b) { return a * b; }
template<typename T>
T ops<T>::div(T a , T b) { return a / b; }
template<typename T>
T ops<T>::pow(T a, T b) { return std::pow(a, b); }
template<typename T>
T ops<T>::eq(T a, T b) { return a == b; }
template<typename T>
T ops<T>::neq(T a, T b) { return a != b; }
template<typename T>
T ops<T>::lt(T a, T b) { return a < b; }
template<typename T>
T ops<T>::le(T a, T b) { return a <= b; }
template<typename T>
T ops<T>::gt(T a, T b) { return a > b; }
template<typename T>
T ops<T>::ge(T a, T b) { return a >= b; }
template<typename T>
T ops<T>::maximum(T a, T b) { if (b > a) { return b; } else { return a; }; }
template<typename T>
T ops<T>::minimum(T a, T b) { if (b < a) { return b; } else { return a; }; }
// elemwise
template<typename T>
void ops<T>::elem(const T* a, const T* b, T fn(T, T), size_t size, T* output) {
    for (size_t i=0; i < size; ++i) {
        output[i] = fn(a[i], b[i]);
    }
}
template<typename T>
void ops<T>::elem(const T* a, T b, T fn(T, T), size_t size, T* output) {
    for (size_t i=0; i < size; ++i) {
        output[i] = fn(a[i], b);
    }
}
template<typename T>
void ops<T>::elem(const T* a, T fn(T), size_t size, T* output) {
    for (size_t i=0; i < size; ++i) {
        output[i] = fn(a[i]);
    }
}
template<typename T>
void ops<T>::elem(const T* a, const T* b, T fn(T, T), size_t size, const T* mask, T* output) {
    for (size_t i=0; i < size; ++i) {
        if ((i > 0) && (mask[i] == 0)) {
            break;
        }
        output[size_t(mask[i])] = fn(a[size_t(mask[i])], b[size_t(mask[i])]);
    }
}
template<typename T>
void ops<T>::elem(const T* a, T b, T fn(T, T), size_t size, const T* mask, T* output) {
    for (size_t i=0; i < size; ++i) {
        if ((i > 0) && (mask[i] == 0)) {
            break;
        }
        output[size_t(mask[i])] = fn(a[size_t(mask[i])], b);
    }
}
template<typename T>
void ops<T>::elem(const T* a, T fn(T), size_t size, const T* mask, T* output) {
    for (size_t i=0; i < size; ++i) {
        if ((i > 0) && (mask[i] == 0)) {
            break;
        }
        output[size_t(mask[i])] = fn(a[size_t(mask[i])]);
    }
}
// slicewise
template<typename T>
void ops<T>::slice(const T* a, size_t start[2], size_t end[2], size_t stride[2],
                   size_t img_shape[2], size_t size, T* output) {
    size_t cnt = 0;
    for (size_t i=0; i < size; i+=(img_shape[0]*img_shape[1])) {
        for (size_t r=start[0]; r < end[0]; r+=stride[0]) {
            for (size_t c=start[1]; c < end[1]; c+=stride[1]) {
                output[cnt] = a[i + r * img_shape[1] + c];
                ++cnt;
            }
        }
    }
}
template<typename T>
size_t ops<T>::slice_index(const T* a, size_t start[2], size_t end[2], size_t stride[2],
                           size_t img_shape[2], size_t size, size_t* output) {
    size_t cnt = 0;
    for (size_t i=0; i < size; i+=(img_shape[0]*img_shape[1])) {
        for (size_t r=start[0]; r < end[0]; r+=stride[0]) {
            for (size_t c=start[1]; c < end[1]; c+=stride[1]) {
                output[cnt] = i + r * img_shape[1] + c;
                ++cnt;
            }
        }
    }
    return cnt;
}
template<typename T>
void ops<T>::slice_put(const T* b, size_t start[2], size_t end[2], size_t stride[2],
                       size_t img_shape[2], size_t size, T* output) {
    size_t cnt = 0;
    for (size_t i=0; i < size; i+=(img_shape[0]*img_shape[1])) {
        for (size_t r=start[0]; r < end[0]; r+=stride[0]) {
            for (size_t c=start[1]; c < end[1]; c+=stride[1]) {
                output[i + r * img_shape[1] + c] = b[cnt];
                ++cnt;
            }
        }
    }
}
template<typename T>
void ops<T>::slice_fn(const T* a, const T* b, T fn(T, T), size_t start[2], size_t end[2],
                      size_t stride[2], size_t img_shape[2], size_t size, T* output) {
    size_t cnt = 0;
    for (size_t i=0; i < size; i+=(img_shape[0]*img_shape[1])) {
        for (size_t r=start[0]; r < end[0]; r+=stride[0]) {
            for (size_t c=start[1]; c < end[1]; c+=stride[1]) {
                size_t idx = i + r * img_shape[1] + c;
                output[idx] = fn(a[idx], b[cnt]);
                ++cnt;
            }
        }
    }
}
template<typename T>
void ops<T>::slice_fn(const T* a, T fn(T), size_t start[2], size_t end[2],
                      size_t stride[2], size_t img_shape[2], size_t size, T* output) {
    size_t cnt = 0;
    for (size_t i=0; i < size; i+=(img_shape[0]*img_shape[1])) {
        for (size_t r=start[0]; r < end[0]; r+=stride[0]) {
            for (size_t c=start[1]; c < end[1]; c+=stride[1]) {
                size_t idx = i + r * img_shape[1] + c;
                output[idx] = fn(a[idx]);
                ++cnt;
            }
        }
    }
}
// kernelwise
template<typename T>
size_t ops<T>::output_size(size_t kernel[2], size_t img_shape[2], size_t stride[2], size_t size) {
    size_t new_size = size / (img_shape[0] * img_shape[1]);
    size_t new_h = (img_shape[0] - kernel[0]) / stride[0] + 1;
    size_t new_w = (img_shape[1] - kernel[1]) / stride[1] + 1;
    return new_size * new_h * new_w;
}
template<typename T>
void ops<T>::kernel_fn(const T* a, T reduce(T, T), size_t kernel[2], size_t img_shape[2],
                       size_t stride[2], size_t size, T* output) {
    size_t block_size = output_size(kernel, img_shape, stride, size);
    T* block = new T[block_size];
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            // get block, apply reduce fn
            slice(a, start, end, stride, img_shape, size, block);
            elem(block, output, reduce, block_size, output);
        }
    }
    delete [] block;
}
template<typename T>
void ops<T>::kernel_fn(const T* a, const T* b, T fn(T, T), T reduce(T, T), size_t kernel[2],
                       size_t img_shape[2], size_t stride[2], size_t size, T* output) {
    size_t block_size = output_size(kernel, img_shape, stride, size);
    T* block = new T[block_size];
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            // get block, apply fn then reduce fn
            slice(a, start, end, stride, img_shape, size, block);
            elem(block, b, fn, block_size, block);
            elem(block, output, reduce, block_size, output);
        }
    }
    delete [] block;
}
template<typename T>
void ops<T>::kernel_fn(const T* a, T b, T fn(T, T), T reduce(T, T), size_t kernel[2],
                       size_t img_shape[2], size_t stride[2], size_t size, T* output) {
    size_t block_size = output_size(kernel, img_shape, stride, size);
    T* block = new T[block_size];
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            // get block, apply fn then reduce fn
            slice(a, start, end, stride, img_shape, size, block);
            elem(block, b, fn, block_size, block);
            elem(block, output, reduce, block_size, output);
        }
    }
    delete [] block;
}
template<typename T>
void ops<T>::kernel_fn(const T* a, T fn(T), T reduce(T, T), size_t kernel[2],
                       size_t img_shape[2], size_t stride[2], size_t size, T* output) {
    size_t block_size = output_size(kernel, img_shape, stride, size);
    T* block = new T[block_size];
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            // get block, apply fn then reduce fn
            slice(a, start, end, stride, img_shape, size, block);
            elem(block, fn, block_size, block);
            elem(block, output, reduce, block_size, output);
        }
    }
    delete [] block;
}
template<typename T>
void ops<T>::kernel_fn(const T* a, const T* b, T fn(T, T), size_t kernel[2],
                       size_t img_shape[2], size_t stride[2], size_t size, T* output) {
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            // apply slice_fn
            slice_fn(a, b, fn, start, end, stride, img_shape, size, output);
        }
    }
}
template<typename T>
void ops<T>::kernel_fn(const T* a, T b, T fn(T, T), size_t kernel[2],
                       size_t img_shape[2], size_t stride[2], size_t size, T* output) {
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            // apply slice_fn
            slice_fn(a, b, fn, start, end, stride, img_shape, size, output);
        }
    }
}
template<typename T>
void ops<T>::kernel_fn(const T* a, T fn(T), size_t kernel[2], size_t img_shape[2],
                       size_t stride[2], size_t size, T* output) {
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            // apply slice_fn
            slice_fn(a, fn, start, end, stride, img_shape, size, output);
        }
    }
}
template<typename T>
T ops<T>::sum(const T* a, size_t size) {
    T o = 0;
    for (size_t i=0; i < size; ++i) {
        o += a[i];
    }
    return o;
}
template<typename T>
T ops<T>::sum(const T* a, size_t size, const T* mask) {
    T o = 0;
    for (size_t i=0; i < size; ++i) {
        if ((i > 0) && (size_t(mask[i]) == 0)) {
            break;
        }
        o += a[size_t(mask[i])];
    }
    return o;
}
template<typename T>
void ops<T>::cumsum(const T* a, size_t size, T* output) {
    T o = 0;
    for (size_t i=0; i < size; ++i) {
        o += a[i];
        output[i] = o;
    }
}
template<typename T>
void ops<T>::cumsum(const T* a, size_t size, const T* mask, T* output) {
    T o = 0;
    for (size_t i=0; i < size; ++i) {
        if ((i > 0) && (size_t(mask[i]) == 0)) {
            break;
        }
        o += a[size_t(mask[i])];
        output[size_t(mask[i])] = o;
    }
}
template<typename T>
void ops<T>::cumsum(const T* a, size_t kernel[2], size_t img_shape[2],
                    size_t stride[2], size_t size, T* output) {
    size_t block_size = output_size(kernel, img_shape, stride, size);
    T* block = new T[block_size];
    T* block_sum = new T[block_size];
    zeros(block_size, block_sum);
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            // add block_sum to current block, then add current block to block_sum
            slice_fn(a, block_sum, add, start, end, stride, img_shape, size, output);
            slice(a, start, end, stride, img_shape, size, block);
            elem(block, block_sum, add, block_size, block_sum);
        }
    }
    delete [] block;
    delete [] block_sum;
}
template<typename T>
T ops<T>::mean(const T* a, size_t size) {
    return sum(a, size) / T(size);
}
template<typename T>
T ops<T>::mean(const T* a, size_t size, const T* mask) {
    return sum(a, size, mask) / T(size);
}
template<typename T>
void ops<T>::softmax(const T* a, bool include_zero, size_t size, T* output) {
    T m = max(a, size);
    elem(a, m, sub, size, output);
    elem(output, exp, size, output);
    T s = sum(output, size);
    if (include_zero) {
        s += 1;
    }
    elem(output, s, div, size, output);
}
template<typename T>
void ops<T>::softmax(const T* a, bool include_zero, size_t size, const T* mask, T* output) {
    T m = max(a, size, mask);
    if (include_zero) {
        m = maximum(m, 0);
    }
    elem(a, m, sub, size, mask, output);
    elem(output, exp, size, mask, output);
    T s = sum(output, size, mask);
    if (include_zero) {
        s += 1;
    }
    elem(output, s, div, size, mask, output);
}
template<typename T>
void ops<T>::softmax(const T* a, bool include_zero, size_t kernel[2], size_t img_shape[2],
                     size_t stride[2], size_t size, T* output) {
    // get max across kernels
    size_t block_size = output_size(kernel, img_shape, stride, size);
    T* block_max = new T[block_size];
    zeros(block_size, block_max);
    kernel_fn(a, maximum, kernel, img_shape, stride, size, block_max);
    // subtract max
    kernel_fn(a, block_max, sub, kernel, img_shape, stride, size, output);
    // exponentiate
    kernel_fn(output, exp, kernel, img_shape, stride, size, output);
    // sum across kernels
    T* block_sum = new T[block_size];
    zeros(block_size, block_sum);
    kernel_fn(output, add, kernel, img_shape, stride, size, block_sum);
    if (include_zero) {
        elem(block_sum, 1, add, block_size, block_sum);
    }
    // divide by sum of exponentials
    kernel_fn(output, block_sum, div, kernel, img_shape, stride, size, output);
    delete [] block_max;
    delete [] block_sum;
}
// utility
template<typename T>
void ops<T>::set(T b, size_t size, T* output) {
    // size set
    for (size_t i=0; i < size; ++i) {
        output[i] = b;
    }
}
template<typename T>
void ops<T>::set(T b, size_t size, const T* mask, T* output) {
    // mask set
    for (size_t i=0; i < size; ++i) {
        if ((i > 0) && (size_t(mask[i]) == 0)) {
            break;
        }
        output[size_t(mask[i])] = b;
    }
}
template<typename T>
void ops<T>::set_array(const T* b, size_t size, T* output) {
    // size set
    for (size_t i=0; i < size; ++i) {
        output[i] = b[i];
    }
}
template<typename T>
void ops<T>::set_array(const T* b, size_t size, const T* mask, T* output) {
    // mask set
    for (size_t i=0; i < size; ++i) {
        if ((i > 0) && (size_t(mask[i]) == 0)) {
            break;
        }
        output[size_t(mask[i])] = b[size_t(mask[i])];
    }
}
template<typename T>
void ops<T>::set_kernel(const T* b, size_t kernel[2], size_t img_shape[2],
                        size_t stride[2], size_t size, T* output) {
    // kernel set
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            // set each block in output to b
            slice_put(b, start, end, stride, img_shape, size, output);
        }
    }
}

template<typename T>
void ops<T>::zeros(size_t size, T* output) {
    set(0, size, output);
}
template<typename T>
void ops<T>::zeros(size_t size, const T* mask, T* output) {
    set(0, size, mask, output);
}
template<typename T>
void ops<T>::ones(size_t size, T* output) {
    set(1, size, output);
}
template<typename T>
void ops<T>::ones(size_t size, const T* mask, T* output) {
    set(1, size, mask, output);
}
template<typename T>
size_t ops<T>::where(const T* a, T b, T fn(T, T), size_t size, size_t* indices) {
    size_t cnt = 0;
    for (size_t i=0; i < size; ++i) {
        if (fn(a[i], b)) {
            indices[cnt] = i;
            ++cnt;
        }
    }
    return cnt;
}
template<typename T>
size_t ops<T>::slice_where(const T* a, const T* b, T fn(T, T), size_t start[2],
                           size_t end[2], size_t stride[2],  size_t img_shape[2],
                           size_t size, T* output, size_t* indices) {
    size_t cnt = 0;
    for (size_t i=0; i < size; i+=(img_shape[0]*img_shape[1])) {
        for (size_t r=start[0]; r < end[0]; r+=stride[0]) {
            for (size_t c=start[1]; c < end[1]; c+=stride[1]) {
                size_t idx = i + r * img_shape[1] + c;
                if (fn(a[idx], b[cnt])) {
                    output[cnt] = a[idx];
                    indices[cnt] = r * img_shape[1] + c;
                }
                ++cnt;
            }
        }
    }
    return cnt;
}
template<typename T>
void ops<T>::find(const T* a, T fn(T, T), size_t size, T& value, size_t& idx)
{
    // size find
    for (size_t i=0; i < size; ++i) {
        if (fn(a[i], value)) {
            value = a[i];
            idx = i;
        }
    }
}
template<typename T>
void ops<T>::find(const T* a, T fn(T, T), size_t size, const T* mask, T& value, size_t& idx)
{
    // mask find
    for (size_t i=0; i < size; ++i) {
        if ((i > 0) && (size_t(mask[i]) == 0)) {
            break;
        } else if (fn(a[size_t(mask[i])], value)) {
            value = a[size_t(mask[i])];
            idx = size_t(mask[i]);
        }
    }
}
template<typename T>
void ops<T>::find(const T* a, T fn(T, T), size_t kernel[2], size_t img_shape[2],
                  size_t stride[2], size_t size, T* values, size_t* indices) {
    // kernel find
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            // perform fn elemwise, find indices where true, set output
            slice_where(a, values, fn, start, end, stride, img_shape, size, values, indices);
        }
    }
}
// max and min
template<typename T>
T ops<T>::max(const T* a, size_t size) {
    T value = -INFINITY;
    size_t idx = 0;
    find(a, gt, size, value, idx);
    return value;
}
template<typename T>
T ops<T>::max(const T* a, size_t size, const T* mask) {
    T value = -INFINITY;
    size_t idx = 0;
    find(a, gt, size, mask, value, idx);
    return value;
}
template<typename T>
void ops<T>::max(const T* a, size_t kernel[2], size_t img_shape[2],
                 size_t stride[2], size_t size, T* output, size_t* indices) {
    size_t block_size = output_size(kernel, img_shape, stride, size);
    set(-INFINITY, block_size, output);
    find(a, gt, kernel, img_shape, stride, size, output, indices);
}
template<typename T>
size_t ops<T>::argmax(const T* a, size_t size) {
    T value = -INFINITY;
    size_t idx = 0;
    find(a, gt, size, value, idx);
    return idx;
}
template<typename T>
size_t ops<T>::argmax(const T* a, size_t size, const T* mask) {
    T value = -INFINITY;
    size_t idx = 0;
    find(a, gt, size, mask, value, idx);
    return idx;
}
template<typename T>
T ops<T>::min(const T* a, size_t size) {
    T value = INFINITY;
    size_t idx = 0;
    find(a, lt, size, value, idx);
    return value;
}
template<typename T>
T ops<T>::min(const T* a, size_t size, const T* mask) {
    T value = INFINITY;
    size_t idx = 0;
    find(a, lt, size, mask, value, idx);
    return value;
}
template<typename T>
void ops<T>::min(const T* a, size_t kernel[2], size_t img_shape[2],
                 size_t stride[2], size_t size, T* output, size_t* indices) {
    size_t block_size = output_size(kernel, img_shape, stride, size);
    set(INFINITY, block_size, output);
    find(a, lt, kernel, img_shape, stride, size, output, indices);
}
template<typename T>
size_t ops<T>::argmin(const T* a, size_t size) {
    T value = INFINITY;
    size_t idx = 0;
    find(a, lt, size, value, idx);
    return idx;
}
template<typename T>
size_t ops<T>::argmin(const T* a, size_t size, const T* mask) {
    T value = INFINITY;
    size_t idx = 0;
    find(a, lt, size, mask, value, idx);
    return idx;
}
template<typename T>
void ops<T>::keep_max(const T* a, size_t size, T* output, size_t* indices) {
    T value = -INFINITY;
    size_t idx = 0;
    find(a, gt, size, value, idx);
    if (value > -INFINITY) {
        output[idx] = value;
        indices[idx] = 1;
    }
}
template<typename T>
void ops<T>::keep_max(const T* a, size_t size, const T* mask, T* output, size_t* indices) {
    T value = -INFINITY;
    size_t idx = 0;
    find(a, gt, size, mask, value, idx);
    if (value > -INFINITY) {
        output[idx] = value;
        indices[idx] = 1;
    }
}
template<typename T>
void ops<T>::keep_max(const T* a, size_t kernel[2], size_t img_shape[2],
                      size_t stride[2], size_t size, T* output, size_t* indices) {
    size_t block_size = output_size(kernel, img_shape, stride, size);
    T* values = new T[block_size];
    set(-INFINITY, block_size, values);
    find(a, gt, kernel, img_shape, stride, size, values, indices);
    for (size_t i=0; i < block_size; ++i) {
        output[indices[i]] = values[i];
    }
    delete [] values;
}
template<typename T>
void ops<T>::set_max(const T* a, T b, size_t size, T* output, size_t* indices) {
    T value = -INFINITY;
    size_t idx = 0;
    find(a, gt, size, value, idx);
    if (value > -INFINITY) {
        output[idx] = b;
        indices[idx] = 1;
    }
}
template<typename T>
void ops<T>::set_max(const T* a, T b, size_t size, const T* mask, T* output, size_t* indices) {
    T value = -INFINITY;
    size_t idx = 0;
    find(a, gt, size, mask, value, idx);
    if (value > -INFINITY) {
        output[idx] = b;
        indices[idx] = 1;
    }
}
template<typename T>
void ops<T>::set_max(const T* a, T b, size_t kernel[2], size_t img_shape[2],
                     size_t stride[2], size_t size, T* output, size_t* indices) {
    size_t block_size = output_size(kernel, img_shape, stride, size);
    T* values = new T[block_size];
    set(-INFINITY, block_size, values);
    find(a, gt, kernel, img_shape, stride, size, values, indices);
    for (size_t i=0; i < block_size; ++i) {
        output[indices[i]] = b;
    }
    delete [] values;
}
