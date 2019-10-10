#pragma once

template<typename T>
struct ops {
    // unary
    static T square(T a);
    static T log(T a);
    static T exp(T a);
    static T sqrt(T a);
    // binary
    static T add(T a, T b);
    static T sub(T a, T b);
    static T mul(T a, T b);
    static T div(T a , T b);
    static T pow(T a, T b);
    static T eq(T a, T b);
    static T neq(T a, T b);
    static T lt(T a, T b);
    static T le(T a, T b);
    static T gt(T a, T b);
    static T ge(T a, T b);
    static T maximum(T a, T b);
    static T minimum(T a, T b);
    // elemwise
    static void elem(const T* a, const T* b, T fn(T, T), size_t size, T* output);
    static void elem(const T* a, T b, T fn(T, T), size_t size, T* output);
    static void elem(const T* a, T fn(T), size_t size, T* output);
    static void elem(const T* a, const T* b, T fn(T, T), size_t size, size_t* mask, T* output);
    static void elem(const T* a, T b, T fn(T, T), size_t size, size_t* mask, T* output);
    static void elem(const T* a, T fn(T), size_t size, size_t* mask, T* output);
    // slicewise
    static void slice(const T* a, size_t start[2], size_t end[2], size_t stride[2],
                      size_t img_shape[2], size_t size, T* output);
    static void slice_fn(const T* a, const T* b, T fn(T, T), size_t start[2], size_t end[2], 
                         size_t stride[2], size_t img_shape[2], size_t size, T* output);
    static void slice_fn(const T* a, T fn(T), size_t start[2], size_t end[2], 
                         size_t stride[2], size_t img_shape[2], size_t size, T* output);
    static size_t slice_index(const T* a, size_t start[2], size_t end[2], size_t stride[2],
                              size_t img_shape[2], size_t size, size_t* output);
    // kernelwise
    static size_t output_size(size_t kernel[2], size_t img_shape[2], size_t stride[2], 
                              size_t size);
    static void kernel_fn(const T* a, T reduce(T, T), size_t kernel[2], size_t img_shape[2],
                          size_t stride[2], size_t size, T* output);
    static void kernel_fn(const T* a, const T* b, T fn(T, T), T reduce(T, T), size_t kernel[2], 
                          size_t img_shape[2], size_t stride[2], size_t size, T* output);
    static void kernel_fn(const T* a, T b, T fn(T, T), T reduce(T, T), size_t kernel[2], 
                          size_t img_shape[2], size_t stride[2], size_t size, T* output);
    static void kernel_fn(const T* a, T fn(T), T reduce(T, T), size_t kernel[2], 
                          size_t img_shape[2], size_t stride[2], size_t size, T* output);
    static void kernel_fn(const T* a, const T* b, T fn(T, T), size_t kernel[2],
                          size_t img_shape[2], size_t stride[2], size_t size, T* output);
    static void kernel_fn(const T* a, T b, T fn(T, T), size_t kernel[2],
                          size_t img_shape[2], size_t stride[2], size_t size, T* output);
    static void kernel_fn(const T* a, T fn(T), size_t kernel[2], size_t img_shape[2],
                          size_t stride[2], size_t size, T* output);
    static T sum(const T* a, size_t size);
    static T sum(const T* a, size_t size, size_t* mask);
    static void cumsum(const T* a, size_t size, T* output);
    static void cumsum(const T* a, size_t size, size_t* mask, T* output);
    static void cumsum(const T* a, size_t kernel[2], size_t img_shape[2],
                       size_t stride[2], size_t size, T* output);
    static T mean(const T* a, size_t size);
    static T mean(const T* a, size_t size, size_t* mask);
    static void softmax(const T* a, bool include_zero, size_t size, T* output);
    static void softmax(const T* a, bool include_zero, size_t size, size_t* mask, T* output);
    static void softmax(const T* a, bool include_zero, size_t kernel[2], size_t img_shape[2],
                        size_t stride[2], size_t size, T* output);
    // utility
    static size_t where(const T* a, T b, T fn(T, T), size_t size, size_t* output);
    static void set_array(const T* b, size_t size, T* output);
    static void set_array(const T* b, size_t size, size_t* mask, T* output);
    static void set(T b, size_t size, T* output);
    static void set(T b, size_t size, size_t* mask, T* output);
    static void zeros(size_t size, T* output);
    static void zeros(size_t size, size_t* mask, T* output);
    static void ones(size_t size, T* output);
    static void ones(size_t size, size_t* mask, T* output);
    static void find(const T* a, T fn(T, T), size_t size, T output[2]);
    static void find(const T* a, T fn(T, T), size_t size, size_t* mask, T output[2]);
    static void find(const T* a, T fn(T, T), size_t kernel[2], size_t img_shape[2],
                     size_t stride[2], size_t size, T* val, size_t* idx);
    // max and min
    static T max(const T* a, size_t size);
    static T max(const T* a, size_t size, size_t* mask);
    static T argmax(const T* a, size_t size);
    static T argmax(const T* a, size_t size, size_t* mask);
    static T min(const T* a, size_t size);
    static T min(const T* a, size_t size, size_t* mask);
    static T argmin(const T* a, size_t size);
    static T argmin(const T* a, size_t size, size_t* mask);
    static void keep_max(const T* a, size_t size, T* output);
    static void keep_max(const T* a, size_t size, size_t* mask, T* output);
    static void keep_max(const T* a, size_t kernel[2], size_t img_shape[2], 
                         size_t stride[2], size_t size, T* output);
    static void set_max(const T* a, T b, size_t size, T* output);
    static void set_max(const T* a, T b, size_t size, size_t* mask, T* output);
    static void set_max(const T* a, T b, size_t kernel[2], size_t img_shape[2], 
                        size_t stride[2], size_t size, T* output);
};