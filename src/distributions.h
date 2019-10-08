#pragma once
#include <random>

template <typename T>
struct distributions {
    // random generator
    std::default_random_engine re;
    static void random_seed(unsigned seed, std::default_random_engine re);
    static std::uniform_real_distribution<T> uniform(T low, T high);
    static void multinomial(const T* a, size_t size, T* output, T rand);
    static void multinomial(const T* a, size_t size, size_t* mask, T* output, T rand);
};