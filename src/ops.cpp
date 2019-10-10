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
void ops<T>::elem(const T* a, const T* b, T fn(T, T), size_t size, size_t* mask, T* output) {
    for (size_t i=0; i < size; ++i) {
        output[mask[i]] = fn(a[mask[i]], b[mask[i]]);
    }
}
template<typename T>
void ops<T>::elem(const T* a, T b, T fn(T, T), size_t size, size_t* mask, T* output) {
    for (size_t i=0; i < size; ++i) {
        output[mask[i]] = fn(a[mask[i]], b);
    }
}
template<typename T>
void ops<T>::elem(const T* a, T fn(T), size_t size, size_t* mask, T* output) {
    for (size_t i=0; i < size; ++i) {
        output[mask[i]] = fn(a[mask[i]]);
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
T ops<T>::sum(const T* a, size_t size, size_t* mask) { 
    T o = 0;
    for (size_t i=0; i < size; ++i) {
        o += a[mask[i]];
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
void ops<T>::cumsum(const T* a, size_t size, size_t* mask, T* output) {
    T o = 0;
    for (size_t i=0; i < size; ++i) {
        o += a[mask[i]];
        output[mask[i]] = o;
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
T ops<T>::mean(const T* a, size_t size, size_t* mask) {
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
void ops<T>::softmax(const T* a, bool include_zero, size_t size, size_t* mask, T* output) {
    T m = max(a, size, mask);
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
size_t ops<T>::where(const T* a, T b, T fn(T, T), size_t size, size_t* output) {
    size_t cnt = 0;
    for (size_t i=0; i < size; ++i) {
        if (fn(a[i], b)) {
            output[cnt] = i;
            ++cnt;
        }
    }
    return cnt;
}
template<typename T>
void ops<T>::set_array(const T* b, size_t size, T* output) {
    // size set
    for (size_t i=0; i < size; ++i) {
        output[i] = b[i];
    }
}
template<typename T>
void ops<T>::set_array(const T* b, size_t size, size_t* mask, T* output) {
    // mask set
    for (size_t i=0; i < size; ++i) {
        output[mask[i]] = b[mask[i]];
    }
}
template<typename T>
void ops<T>::set(T b, size_t size, T* output) {
    // size set
    for (size_t i=0; i < size; ++i) {
        output[i] = b;
    }
}
template<typename T>
void ops<T>::set(T b, size_t size, size_t* mask, T* output) {
    // mask set
    for (size_t i=0; i < size; ++i) {
        output[mask[i]] = b;
    }
}
template<typename T>
void ops<T>::zeros(size_t size, T* output) {
    set(0, size, output);
}
template<typename T>
void ops<T>::zeros(size_t size, size_t* mask, T* output) {
    set(0, size, mask, output);
}
template<typename T>
void ops<T>::ones(size_t size, T* output) {
    set(1, size, output);
}
template<typename T>
void ops<T>::ones(size_t size, size_t* mask, T* output) {
    set(1, size, mask, output);
}
template<typename T>
void ops<T>::find(const T* a, T fn(T, T), size_t size, T output[2]) 
{
    // size find
    for (size_t i=0; i < size; ++i) {
        if (fn(a[i], output[0])) {
            output[0] = a[i];
            output[1] = T(i);
        }
    }
}
template<typename T>
void ops<T>::find(const T* a, T fn(T, T), size_t size, size_t* mask, T output[2]) 
{
    // mask find
    for (size_t i=0; i < size; ++i) {
        if (fn(a[mask[i]], output[0])) {
            output[0] = a[mask[i]];
            output[1] = T(mask[i]);
        }
    }
}
template<typename T>
void ops<T>::find(const T* a, T fn(T, T), size_t kernel[2], size_t img_shape[2],
                  size_t stride[2], size_t size, T* val, size_t* idx) {
    // kernel find
    size_t block_size = output_size(kernel, img_shape, stride, size);
    T* block = new T[block_size];
    size_t* block_index = new size_t[block_size];
    T* block_eval = new T[block_size];
    size_t* block_mask = new size_t[block_size];
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            // perform fn elemwise, find indices where true, set output
            slice(a, start, end, stride, img_shape, size, block);
            slice_index(a, start, end, stride, img_shape, size, block_index);
            elem(block, val, fn, block_size, block_eval);
            size_t mask_size = where(block_eval, 0, gt, block_size, block_mask);
            set_array(block, mask_size, block_mask, val);
            ops<size_t>::set_array(block_index, mask_size, block_mask, idx);
        }
    }
    delete [] block;
    delete [] block_index;
    delete [] block_eval;
    delete [] block_mask;
}
// max and min
template<typename T>
T ops<T>::max(const T* a, size_t size) {
    T output[2] = {-INFINITY, -1};
    find(a, gt, size, output);
    return output[0];
}
template<typename T>
T ops<T>::max(const T* a, size_t size, size_t* mask) {
    T output[2] = {-INFINITY, -1};
    find(a, gt, size, mask, output);
    return output[0];
}
template<typename T>
T ops<T>::argmax(const T* a, size_t size) {
    T output[2] = {-INFINITY, -1};
    find(a, gt, size, output);
    return output[1];
}
template<typename T>
T ops<T>::argmax(const T* a, size_t size, size_t* mask) {
    T output[2] = {-INFINITY, -1};
    find(a, gt, size, mask, output);
    return output[1];
}
template<typename T>
T ops<T>::min(const T* a, size_t size) {
    T output[2] = {INFINITY, -1};
    find(a, lt, size, output);
    return output[0];
}
template<typename T>
T ops<T>::min(const T* a, size_t size, size_t* mask) {
    T output[2] = {INFINITY, -1};
    find(a, lt, size, mask, output);
    return output[0];
}
template<typename T>
T ops<T>::argmin(const T* a, size_t size) {
    T output[2] = {INFINITY, -1};
    find(a, lt, size, output);
    return output[1];
}
template<typename T>
T ops<T>::argmin(const T* a, size_t size, size_t* mask) {
    T output[2] = {INFINITY, -1};
    find(a, lt, size, mask, output);
    return output[1];
}
template<typename T>
void ops<T>::keep_max(const T* a, size_t size, T* output) {
    T o[2] = {-INFINITY, -1};
    find(a, gt, size, o);
    zeros(size, output);
    if ((o[1] > -1) & (output[size_t(o[1])] != o[0]))
        output[size_t(o[1])] = o[0];
}
template<typename T>
void ops<T>::keep_max(const T* a, size_t size, size_t* mask, T* output) {
    T o[2] = {-INFINITY, -1};
    find(a, gt, size, mask, o);
    zeros(size, mask, output);
    if ((o[1] > -1) & (output[size_t(o[1])] != o[0]))
        output[size_t(o[1])] = o[0];
}
template<typename T>
void ops<T>::keep_max(const T* a, size_t kernel[2], size_t img_shape[2], 
                      size_t stride[2], size_t size, T* output) {
    size_t block_size = output_size(kernel, img_shape, stride, size);
    T* value = new T[block_size];
    size_t* idx = new size_t[block_size];
    set(-INFINITY, block_size, value);
    find(a, gt, kernel, img_shape, stride, size, value, idx);
    zeros(size, output);
    for (size_t i=0; i < block_size; ++i) {
        output[idx[i]] = value[i];
    }
    delete [] value;
    delete [] idx;
}
template<typename T>
void ops<T>::set_max(const T* a, T b, size_t size, T* output) {
    T o[2] = {-INFINITY, -1};
    find(a, gt, size, o);
    zeros(size, output);
    if ((o[1] > -1) & (output[size_t(o[1])] != b))
        output[size_t(o[1])] = b;
}
template<typename T>
void ops<T>::set_max(const T* a, T b, size_t size, size_t* mask, T* output) {
    T o[2] = {-INFINITY, -1};
    find(a, gt, size, mask, o);
    zeros(size, mask, output);
    if ((o[1] > -1) & (output[size_t(o[1])] != b))
        output[size_t(o[1])] = b;
}
template<typename T>
void ops<T>::set_max(const T* a, T b, size_t kernel[2], size_t img_shape[2], 
                     size_t stride[2], size_t size, T* output) {
    size_t block_size = output_size(kernel, img_shape, stride, size);
    T* value = new T[block_size];
    size_t* idx = new size_t[block_size];
    set(-INFINITY, block_size, value);
    find(a, gt, kernel, img_shape, stride, size, value, idx);
    zeros(size, output);
    set(b, block_size, idx, output);
    delete [] value;
    delete [] idx;
}