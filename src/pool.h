#pragma once

template<typename T>
struct pool {
    static void rf_max_pool(const T* array, size_t size, const T* mask, T* output);
    static void rf_avg_pool(const T* array, size_t size, const T* mask, T* output);
    static void rf_sum_pool(const T* array, size_t size, const T* mask, T* output);
    static void rf_probmax_pool(const T* array, size_t size, const T* mask, T* output);
    static void rf_stochastic_pool(const T* array, size_t size, const T* mask, T* output);
    static void rf_lp_pool(const T* array, T p, size_t size, const T* mask, T* output);
    static void kernel_max_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                                size_t stride[2], size_t size, T* output);
    static void kernel_avg_pool(const T* array, size_t kernel[2], size_t img_shape[2],
                                 size_t stride[2], size_t size, T* output);
    static void kernel_sum_pool(const T* array, size_t kernel[2], size_t img_shape[2],
                                size_t stride[2], size_t size, T* output);
    static void kernel_probmax(const T* array, size_t kernel[2], size_t img_shape[2],
                               size_t stride[2], size_t size, T* output);
    static void kernel_probmax_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                                    size_t stride[2], size_t size, T* output);
    static void kernel_stochastic_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                                       size_t stride[2], size_t size, T* output);
    static void kernel_lp_pool(const T* array, T p, size_t kernel[2], size_t img_shape[2], 
                               size_t stride[2], size_t size, T* output);
};
