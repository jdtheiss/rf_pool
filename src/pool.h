#pragma once

template<typename T>
struct pool {
    static void rf_max_pool(const T* array, const T* mask, size_t size, const T* mask_indices, 
                            T* output, size_t* indices, bool apply_mask = false);
    static void rf_probmax_pool(const T* array, const T* mask, size_t size, const T* mask_indices,
                                T* output, size_t* indices, bool apply_mask = false);
    static void rf_stochastic_pool(const T* array, const T* mask, size_t size, const T* mask_indices,
                                   T* output, size_t* indices, bool apply_mask = false);
    static void rf_lp_pool(const T* array, T p, const T* mask, size_t size, const T* mask_indices,
                           T* output, size_t* indices, bool apply_mask = false);
    static void kernel_max_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                                size_t stride[2], size_t size, T* output, size_t* indices);
    static void kernel_probmax(const T* array, size_t kernel[2], size_t img_shape[2],
                               size_t stride[2], size_t size, T* output, size_t* indices);
    static void kernel_probmax_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                                    size_t stride[2], size_t size, T* output, size_t* indices);
    static void kernel_stochastic_pool(const T* array, size_t kernel[2], size_t img_shape[2], 
                                       size_t stride[2], size_t size, T* output, size_t* indices);
    static void kernel_lp_pool(const T* array, T p, size_t kernel[2], size_t img_shape[2], 
                               size_t stride[2], size_t size, T* output, size_t* indices);
    static void kernel_unpool(const T* array, size_t kernel[2], size_t img_shape[2], 
                              size_t stride[2], size_t size, const T* mask, T* output);
};
