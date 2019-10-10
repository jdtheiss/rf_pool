#pragma once

template<typename T>
struct distributions {
    static T rand();
    static void multinomial(const T* a, size_t size, T* output);
    static void multinomial(const T* a, size_t size, size_t* mask, T* output);
    static void multinomial(const T* a, size_t kernel[2], size_t img_shape[2], 
                            size_t stride[2], size_t size, T* output);
};