#pragma once
#include "ops.cpp"
#include "distributions.h"

template<typename T>
T distributions<T>::rand() { return T(std::rand() % 10000) / 10000; }
template<typename T>
void distributions<T>::multinomial(const T* a, size_t size, T* output, size_t* indices) {
    ops<T>::cumsum(a, size, output);
    ops<T>::elem(output, rand(), ops<T>::gt, size, output);
    ops<T>::set_max(output, 1, size, output, indices);
}
template<typename T>
void distributions<T>::multinomial(const T* a, size_t size, const T* mask, T* output, size_t* indices) {
    ops<T>::cumsum(a, size, mask, output);
    ops<T>::elem(output, rand(), ops<T>::gt, size, mask, output);
    ops<T>::set_max(output, 1, size, mask, output, indices);
}
template<typename T>
void distributions<T>::multinomial(const T* a, size_t kernel[2], size_t img_shape[2], 
                                   size_t stride[2], size_t size, T* output, size_t* indices) {
    size_t block_size = ops<T>::output_size(kernel, img_shape, stride, size);
    T* r = new T[block_size];
    for (size_t i=0; i < block_size; ++i) {
        r[i] = rand();
    }
    ops<T>::cumsum(a, kernel, img_shape, stride, size, output);
    ops<T>::kernel_fn(output, r, ops<T>::gt, kernel, img_shape, stride, size, output);
    ops<T>::set_max(output, 1, kernel, img_shape, stride, size, output, indices);
    delete [] r;
}