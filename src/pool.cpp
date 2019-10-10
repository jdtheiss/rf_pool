#pragma once
#include "distributions.cpp"
#include "pool.h"

// max pool operation, set output to max value in mask at max index
template<typename T>
void pool<T>::rf_max_pool(const T* array, size_t size, const T* mask, T* output) {
    size_t* mask_index = new size_t[size];
    size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, size, mask_index);
    ops<T>::keep_max(array, index_size, mask_index, output);
    delete [] mask_index;
}
// avg pool operation, set output to average value in mask at max index
template<typename T>
void pool<T>::rf_avg_pool(const T* array, size_t size, const T* mask, T* output) {
    size_t* mask_index = new size_t[size];
    size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, size, mask_index);
    T m = ops<T>::mean(array, index_size, mask_index);
    ops<T>::set_max(array, m, index_size, mask_index, output);
    delete [] mask_index;
}
// sum pool operation, set output to sum across mask at max index
template<typename T>
void pool<T>::rf_sum_pool(const T* array, size_t size, const T* mask, T* output) {
    size_t* mask_index = new size_t[size];
    size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, size, mask_index);
    T s = ops<T>::sum(array, index_size, mask_index);
    ops<T>::set_max(array, s, index_size, mask_index, output);
    delete [] mask_index;
}
// probmax pool operation, set output to (1-prob all pixels off) across mask at multinomial index
template<typename T>
void pool<T>::rf_probmax_pool(const T* array, size_t size, const T* mask, T* output) {
    size_t* mask_index = new size_t[size];
    size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, size, mask_index);
    ops<T>::softmax(array, true, index_size, mask_index, output);
    T p_sum = ops<T>::sum(output, index_size, mask_index);
    distributions<T>::multinomial(output, index_size, mask_index, output);
    ops<T>::elem(output, p_sum, ops<T>::mul, index_size, mask_index, output);
    delete [] mask_index;
}
// stochastic max pool operation, set output to value at sampled index in mask
template<typename T>
void pool<T>::rf_stochastic_pool(const T* array, size_t size, const T* mask, T* output) {
    size_t* mask_index = new size_t[size];
    size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, size, mask_index);
    ops<T>::softmax(array, false, index_size, mask_index, output);
    distributions<T>::multinomial(output, index_size, mask_index, output);
    ops<T>::elem(array, output, ops<T>::mul, index_size, mask_index, output);
    delete [] mask_index;
}
// lp pool operation, set output to LP value across mask at max index
template<typename T>
void pool<T>::rf_lp_pool(const T* array, T p, size_t size, const T* mask, T* output) {
    size_t* mask_index = new size_t[size];
    size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, size, mask_index);
    ops<T>::elem(array, p, ops<T>::pow, index_size, mask_index, output);
    T sum_output = ops<T>::sum(output, index_size, mask_index);
    ops<T>::set_max(array, ops<T>::pow(sum_output, 1 / p), index_size, mask_index, output);
    delete [] mask_index;
}
// max pool with kernel, set output to max value across kernel blocks
template<typename T>
void pool<T>::kernel_max_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                            size_t stride[2], size_t size, T* output) {
    ops<T>::kernel_fn(array, ops<T>::maximum, kernel, img_shape, stride, size, output);
}
// avg pool with kernel, set output to average across kernel blocks
template<typename T>
void pool<T>::kernel_avg_pool(const T* array, size_t kernel[2], size_t img_shape[2],
                             size_t stride[2], size_t size, T* output) {
    T n = T(kernel[0] * kernel[1]);
    ops<T>::kernel_fn(array, n, ops<T>::div, ops<T>::add, kernel, img_shape, stride, size, output);
}
// sum pool with kernel, set output to sum across kernel blocks
template<typename T>
void pool<T>::kernel_sum_pool(const T* array, size_t kernel[2], size_t img_shape[2],
                              size_t stride[2], size_t size, T* output) {
    ops<T>::kernel_fn(array, ops<T>::add, kernel, img_shape, stride, size, output);
}
// probmax with kernel, set output to (1-prob all pixels off) across kernels at multinomial index
template<typename T>
void pool<T>::kernel_probmax(const T* array, size_t kernel[2], size_t img_shape[2], 
                             size_t stride[2], size_t size, T* output) {
    T* p = new T[size];
    ops<T>::softmax(array, true, kernel, img_shape, stride, size, p);
    T* mult_output = new T[size];
    ops<T>::zeros(size, mult_output);
    distributions<T>::multinomial(p, kernel, img_shape, stride, size, mult_output);
    size_t block_size = ops<T>::output_size(kernel, img_shape, stride, size);
    T* p_sum = new T[block_size];
    ops<T>::zeros(block_size, p_sum);
    ops<T>::kernel_fn(p, ops<T>::add, kernel, img_shape, stride, size, p_sum);
    ops<T>::kernel_fn(mult_output, p_sum, ops<T>::mul, kernel, img_shape, stride, size, output);
    delete [] p;
    delete [] mult_output;
    delete [] p_sum;
}
// probmax pool with kernel, set output to (1-prob all pixels off) across kernels
template<typename T>
void pool<T>::kernel_probmax_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                                  size_t stride[2], size_t size, T* output) {
    T* soft_output = new T[size];
    ops<T>::zeros(size, soft_output);
    kernel_probmax(array, kernel, img_shape, stride, size, soft_output);
    ops<T>::kernel_fn(soft_output, ops<T>::add, kernel, img_shape, stride, size, output);
    delete [] soft_output;
}
// probmax with kernel, set output to (1-prob all pixels off) across kernels at multinomial index
template<typename T>
void pool<T>::kernel_stochastic_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                                     size_t stride[2], size_t size, T* output) {
    T* p = new T[size];
    ops<T>::softmax(array, false, kernel, img_shape, stride, size, p);
    distributions<T>::multinomial(p, kernel, img_shape, stride, size, p);
    ops<T>::elem(array, p, ops<T>::mul, size, p);
    ops<T>::kernel_fn(p, ops<T>::add, kernel, img_shape, stride, size, output);
    delete [] p;
}
// lp pool with kernel, set output to LP value across kernel blocks
template<typename T>
void pool<T>::kernel_lp_pool(const T* array, T p, size_t kernel[2], size_t img_shape[2], 
                           size_t stride[2], size_t size, T* output) {
    // raise each to power p, sum across kernels, get root p
    size_t block_size = ops<T>::output_size(kernel, img_shape, stride, size);
    ops<T>::kernel_fn(array, p, ops<T>::pow, ops<T>::add, kernel, img_shape, stride, size, output);
    ops<T>::elem(output, 1 / p, ops<T>::pow, block_size, output);
}