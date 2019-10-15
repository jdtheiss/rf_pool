#pragma once
#include "distributions.cpp"
#include "pool.h"

// max pool operation, set output to max value in mask_indices at max index
template<typename T>
void pool<T>::rf_max_pool(const T* array, size_t size, const T* mask_indices, T* output, size_t* indices) {
    ops<T>::keep_max(array, size, mask_indices, output, indices);
}
// probmax pool operation, set output to (1-prob all pixels off) across mask_indices at multinomial index
template<typename T>
void pool<T>::rf_probmax_pool(const T* array, size_t size, const T* mask_indices, T* output, size_t* indices) {
    ops<T>::softmax(array, true, size, mask_indices, output);
    T p_sum = ops<T>::sum(output, size, mask_indices);
    distributions<T>::multinomial(output, size, mask_indices, output, indices);
    ops<T>::elem(output, p_sum, ops<T>::mul, size, mask_indices, output);
}
// stochastic max pool operation, set output to value at multinomial index in mask_indices
template<typename T>
void pool<T>::rf_stochastic_pool(const T* array, size_t size, const T* mask_indices, T* output, size_t* indices) {
    ops<T>::softmax(array, false, size, mask_indices, output);
    distributions<T>::multinomial(output, size, mask_indices, output, indices);
    ops<T>::elem(array, output, ops<T>::mul, size, mask_indices, output);
}
// lp pool operation, set output to LP value across mask_indices at max index
template<typename T>
void pool<T>::rf_lp_pool(const T* array, T p, size_t size, const T* mask_indices, T* output, size_t* indices) {
    ops<T>::elem(array, p, ops<T>::pow, size, mask_indices, output);
    T sum_output = ops<T>::sum(output, size, mask_indices);
    ops<T>::set_max(array, ops<T>::pow(sum_output, 1 / p), size, mask_indices, output, indices);
}
// max pool with kernel, set output to max value across kernel blocks
template<typename T>
void pool<T>::kernel_max_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                              size_t stride[2], size_t size, T* output, size_t* indices) {
    ops<T>::max(array, kernel, img_shape, stride, size, output, indices);
}
// probmax with kernel, set output to (1-prob all pixels off) across kernels at multinomial index
template<typename T>
void pool<T>::kernel_probmax(const T* array, size_t kernel[2], size_t img_shape[2], 
                             size_t stride[2], size_t size, T* output, size_t* indices) {
    T* p = new T[size];
    ops<T>::softmax(array, true, kernel, img_shape, stride, size, p);
    distributions<T>::multinomial(p, kernel, img_shape, stride, size, output, indices);
    size_t block_size = ops<T>::output_size(kernel, img_shape, stride, size);
    T* p_sum = new T[block_size];
    ops<T>::zeros(block_size, p_sum);
    ops<T>::kernel_fn(p, ops<T>::add, kernel, img_shape, stride, size, p_sum);
    ops<T>::kernel_fn(output, p_sum, ops<T>::mul, kernel, img_shape, stride, size, output);
    delete [] p;
    delete [] p_sum;
}
// probmax pool with kernel, set output to (1-prob all pixels off) across kernels
template<typename T>
void pool<T>::kernel_probmax_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                                  size_t stride[2], size_t size, T* output, size_t* indices) {
    T* soft_output = new T[size];
    ops<T>::zeros(size, soft_output);
    kernel_probmax(array, kernel, img_shape, stride, size, soft_output, indices);
    ops<T>::kernel_fn(soft_output, ops<T>::add, kernel, img_shape, stride, size, output);
    delete [] soft_output;
}
// probmax with kernel, set output to (1-prob all pixels off) across kernels at multinomial index
template<typename T>
void pool<T>::kernel_stochastic_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                                     size_t stride[2], size_t size, T* output, size_t* indices) {
    T* p = new T[size];
    ops<T>::softmax(array, false, kernel, img_shape, stride, size, p);
    distributions<T>::multinomial(p, kernel, img_shape, stride, size, p, indices);
    ops<T>::elem(array, p, ops<T>::mul, size, p);
    ops<T>::kernel_fn(p, ops<T>::add, kernel, img_shape, stride, size, output);
    delete [] p;
}
// lp pool with kernel, set output to LP value across kernel blocks
template<typename T>
void pool<T>::kernel_lp_pool(const T* array, T p, size_t kernel[2], size_t img_shape[2], 
                             size_t stride[2], size_t size, T* output, size_t* indices) {
    // raise each to power p, sum across kernels, get root p
    size_t block_size = ops<T>::output_size(kernel, img_shape, stride, size);
    ops<T>::kernel_fn(array, p, ops<T>::pow, ops<T>::add, kernel, img_shape, stride, size, output);
    ops<T>::elem(output, 1 / p, ops<T>::pow, block_size, output);
}
// unpool with kernel, set output to values at indices in resized array
template<typename T>
void pool<T>::kernel_unpool(const T* array, size_t kernel[2], size_t img_shape[2], 
                            size_t stride[2], size_t size, const T* mask, T* output) {
    ops<T>::set_kernel(array, kernel, img_shape, stride, size, output);
    ops<T>::elem(output, mask, ops<T>::mul, size, output);
}