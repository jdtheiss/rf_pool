#pragma once
#include "distributions.h"
#include "ops.cpp"

template <typename T>
void distributions<T>::random_seed(unsigned seed, std::default_random_engine re) {
    re.seed(seed);
}
template <typename T>
std::uniform_real_distribution<T> distributions<T>::uniform(T low, T high) {
    std::uniform_real_distribution<T> distr(low, high);
    return distr;
}
template <typename T>
void distributions<T>::multinomial(const T* a, size_t size, T* output,
                                   T rand) {
    ops<T>::cumsum(a, size, output);
    ops<T>::elem(output, rand, ops<T>::gt, size, output);
    ops<T>::keep_max(output, size, output);
}
template <typename T>
void distributions<T>::multinomial(const T* a, size_t size, size_t* mask, T* output,
                                   T rand) {
    ops<T>::cumsum(a, size, mask, output);
    ops<T>::elem(output, rand, ops<T>::gt, size, mask, output);
    ops<T>::keep_max(output, size, mask, output);
}