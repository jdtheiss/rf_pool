#pragma once
#include "pool.h"
#include "ops.cpp"
#include "distributions.cpp"

// set distributions struct for multinomial
static distributions<float> distr;
// get output_size of kernel pool
template<typename T>
size_t pool<T>::output_size(size_t size, size_t img_shape[2], size_t kernel[2], size_t stride[2]) {
    size_t new_size = size / (img_shape[0] * img_shape[1]);
    size_t new_h = (img_shape[0] - kernel[0]) / stride[0] + 1;
    size_t new_w = (img_shape[1] - kernel[1]) / stride[1] + 1;
    return new_size * new_h * new_w;
}
// max pool operation, set output to max value in mask at max index
template<typename T>
void pool<T>::rf_max_pool(const T* array, size_t mask_size, const T* mask, T* output) {
    size_t* mask_index = (size_t*) malloc(mask_size * sizeof(size_t));
    size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, mask_size, mask_index);
    ops<T>::keep_max(array, index_size, mask_index, output);
    free(mask_index);
}
// avg pool operation, set output to average value in mask at max index
template<typename T>
void pool<T>::rf_avg_pool(const T* array, size_t mask_size, const T* mask, T* output) {
    size_t* mask_index = (size_t*) malloc(mask_size * sizeof(size_t));
    size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, mask_size, mask_index);
    T m = ops<T>::mean(array, index_size, mask_index);
    ops<T>::set_max(array, m, index_size, mask_index, output);
    free(mask_index);
}
// sum pool operation, set output to sum across mask at max index
template<typename T>
void pool<T>::rf_sum_pool(const T* array, size_t mask_size, const T* mask, T* output) {
    size_t* mask_index = (size_t*) malloc(mask_size * sizeof(size_t));
    size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, mask_size, mask_index);
    T s = ops<T>::sum(array, index_size, mask_index);
    ops<T>::set_max(array, s, index_size, mask_index, output);
    free(mask_index);
}
// softmax pool operation, set output to softmax across mask
template<typename T>
void pool<T>::rf_softmax(const T* array, size_t mask_size, const T* mask, T* output) {
    size_t* mask_index = (size_t*) malloc(mask_size * sizeof(size_t));
    size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, mask_size, mask_index);
    ops<T>::softmax(array, true, index_size, mask_index, output);
    free(mask_index);
}
// softmax pool operation, set output to softmax across mask
template<typename T>
void pool<T>::rf_soft_pool(const T* array, size_t mask_size, const T* mask, T* output) {
    size_t* mask_index = (size_t*) malloc(mask_size * sizeof(size_t));
    size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, mask_size, mask_index);
    ops<T>::softmax(array, true, index_size, mask_index, output);
    T p = ops<T>::sum(output, index_size, mask_index);
    distr.multinomial(output, index_size, mask_index, output,
                      distr.uniform(0,1)(distr.re));
    if (ops<T>::sum(output, index_size, mask_index) > 0) {
        ops<T>::set_max(output, p, index_size, mask_index, output);
    }
    free(mask_index);
}
// stochastic max pool operation, set output to value at sampled index in mask
template<typename T>
    void pool<T>::rf_stochastic_pool(const T* array, size_t mask_size, const T* mask, T* output) {
        size_t* mask_index = (size_t*) malloc(mask_size * sizeof(size_t));
        size_t index_size = ops<T>::where(mask, 0, ops<T>::gt, mask_size, mask_index);
        ops<T>::softmax(array, false, index_size, mask_index, output);
        ops<T>::multinomial(output, index_size, mask_index, output);
        ops<T>::elem(array, output, ops<T>::mul, index_size, mask_index, output);
        free(mask_index);
    }
// fn pool with kernel, set output to fn value across kernel blocks
template<typename T>
void pool<T>::kernel_pool(const T* array, T fn(T, T), size_t kernel[2], size_t img_shape[2], 
                        size_t stride[2], size_t size, T* output) {
    size_t block_size = output_size(size, img_shape, kernel, stride);
    T* block = (T*) malloc(block_size * sizeof(T));
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            ops<T>::slice(array, start, end, stride, img_shape, size, block);
            ops<T>::elem(block, output, fn, block_size, output);
        }
    }
    free(block);
}
// max pool with kernel, set output to max value across kernel blocks
template<typename T>
void pool<T>::kernel_max_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                            size_t stride[2], size_t size, T* output) {
    kernel_pool(array, ops<T>::maximum, kernel, img_shape, stride, size, output);
}
// avg pool with kernel, set output to average across kernel blocks
template<typename T>
void pool<T>::kernel_avg_pool(const T* array, size_t kernel[2], size_t img_shape[2],
                             size_t stride[2], size_t size, T* output) {
    kernel_pool(array, ops<T>::add, kernel, img_shape, stride, size, output);
    size_t block_size = output_size(size, img_shape, kernel, stride);
    T n = T(kernel[0] * kernel[1]);
    ops<T>::elem(output, n, ops<T>::div, block_size, output);
}
// sum pool with kernel, set output to sum across kernel blocks
template<typename T>
void pool<T>::kernel_sum_pool(const T* array, size_t kernel[2], size_t img_shape[2],
                            size_t stride[2], size_t size, T* output) {
    kernel_pool(array, ops<T>::add, kernel, img_shape, stride, size, output);
}
// lp pool with kernel, set output to LP value across kernel blocks
template<typename T>
void pool<T>::kernel_lp_pool(const T* array, T p, size_t kernel[2], size_t img_shape[2], 
                           size_t stride[2], size_t size, T* output) {
    size_t block_size = output_size(size, img_shape, kernel, stride);
    T* block = (T*) malloc(block_size * sizeof(T));
    size_t start[] = {0, 0};
    size_t end[] = {img_shape[0], img_shape[1]};
    for (size_t r=0; r < kernel[0]; ++r) {
        start[0] = r;
        end[0] = img_shape[0] - (kernel[0] - 1 - r);
        for (size_t c=0; c < kernel[1]; ++c) {
            start[1] = c;
            end[1] = img_shape[1] - (kernel[1] - 1 - c);
            ops<T>::slice(array, start, end, stride, img_shape, size, block);
            ops<T>::elem(block, p, ops<T>::pow, block_size, block);
            ops<T>::elem(block, output, ops<T>::add, block_size, output);
        }
    }
    ops<T>::elem(output, 1 / p, ops<T>::pow, block_size, output);
    free(block);
}