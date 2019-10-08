#pragma once
#include "ops.h"
#include <cmath>

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
void ops<T>::elem(const T* a, T* b, T fn(T, T), size_t size, size_t* mask, T* output) {
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
    if (include_zero) 
        s += 1;
    elem(output, s, div, size, output);
}
template<typename T>
void ops<T>::softmax(const T* a, bool include_zero, size_t size, size_t* mask, T* output) {
    T m = max(a, size, mask);
    elem(a, m, sub, size, mask, output);
    elem(output, exp, size, mask, output);
    T s = sum(output, size, mask);
    if (include_zero) 
        s += 1;
    elem(output, s, div, size, mask, output);
}
//div_norm
// utility
template<typename T>
size_t ops<T>::where(const T* a, T b, T fn(T, T), size_t size, size_t* output) {
    size_t cnt = 0;
    for (size_t i=0; i < size; ++i) {
        if (fn(a[i], b) == 1) {
            output[cnt] = i;
            ++cnt;
        }
    }
    return cnt;
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
void ops<T>::slice(const T* a, size_t start[2], size_t end[2], size_t stride[2],
                  size_t img_shape[2], size_t size, T* output) {
    size_t cnt = 0;
    for (size_t i=0; i < size; i+=(img_shape[0]*img_shape[1])) {
        for (size_t r=start[0]; r < end[0]; r+=stride[0]) {
            for (size_t c=start[1]; c < end[1]; c+=stride[1]) {
                output[cnt] = a[i + r*img_shape[1] + c];
                ++cnt;
            }
        }
    }
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
    if ((o[1] > -1) & (output[size_t(o[1])] != o[0]))
        output[size_t(o[1])] = o[0];
}
template<typename T>
void ops<T>::keep_max(const T* a, size_t size, size_t* mask, T* output) {
    T o[2] = {-INFINITY, -1};
    find(a, gt, size, mask, o);
    if ((o[1] > -1) & (output[size_t(o[1])] != o[0]))
        output[size_t(o[1])] = o[0];
}
template<typename T>
void ops<T>::set_max(const T* a, T b, size_t size, T* output) {
    T o[2] = {-INFINITY, -1};
    find(a, gt, size, o);
    if ((o[1] > -1) & (output[size_t(o[1])] != b))
        output[size_t(o[1])] = b;
}
template<typename T>
void ops<T>::set_max(const T* a, T b, size_t size, size_t* mask, T* output) {
    T o[2] = {-INFINITY, -1};
    find(a, gt, size, mask, o);
    if ((o[1] > -1) & (output[size_t(o[1])] != b))
        output[size_t(o[1])] = b;
}