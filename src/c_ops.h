#include <cmath>

template <typename T>
class c_ops {
public:
    // unary
    static T square(T a) { return a * a; }
    static T log(T a) { return std::log(a); }
    static T exp(T a) { return std::exp(a); }
    static T sqrt(T a) { return std::sqrt(a); }
    // binary
    static T add(T a, T b) { return a + b; }
    static T sub(T a, T b) { return a - b; }
    static T mul(T a, T b) { return a * b; }
    static T div(T a , T b) { return a / b; }
    static T pow(T a, T b) { return std::pow(a, b); }
    static T eq(T a, T b) { return a == b; }
    static T neq(T a, T b) { return a != b; }
    static T lt(T a, T b) { return a < b; }
    static T le(T a, T b) { return a <= b; }
    static T gt(T a, T b) { return a > b; }
    static T ge(T a, T b) { return a >= b; }
    static T max(T a, T b) { if (b > a) { return b; } else { return a; }; }
    static T min(T a, T b) { if (b < a) { return b; } else { return a; }; }
    // elemwise
    static void elem(const T* a, const T* b, T fn(T, T), size_t size, T* output) { 
        for (size_t i=0; i < size; ++i) {
            output[i] = fn(a[i], b[i]);
        }
    }
    static void elem(const T* a, T b, T fn(T, T), size_t size, T* output) {
        for (size_t i=0; i < size; ++i) {
            output[i] = fn(a[i], b);
        }
    }
    static size_t where(const T* a, T b, T fn(T, T), size_t size, size_t* output) {
        size_t cnt = 0;
        for (size_t i=0; i < size; ++i) {
            if (fn(a[i], b) == 1) {
                output[cnt] = i;
                ++cnt;
            }
        }
        return cnt;
    }
    static void mask_fn(const T* a, T* b, T fn(T, T), size_t size, T* output, size_t* mask) {
        for (size_t i=0; i < size; ++i) {
            output[mask[i]] = fn(a[mask[i]], b[mask[i]]);
        }
    }
    static void mask_fn(const T* a, T b, T fn(T, T), size_t size, T* output, size_t* mask) {
        for (size_t i=0; i < size; ++i) {
            output[mask[i]] = fn(a[mask[i]], b);
        }
    }
    static void mask_fn(const T* a, T fn(T), size_t size, T* output, size_t* mask) {
        for (size_t i=0; i < size; ++i) {
            output[mask[i]] = fn(a[mask[i]]);
        }
    }
    // utility
    static void set(T b, size_t size, T* output) {
        // size set
        for (size_t i=0; i < size; ++i) {
            output[i] = b;
        }
    }
    static void set(T b, size_t size, size_t* mask, T* output) {
        // mask set
        for (size_t i=0; i < size; ++i) {
            output[mask[i]] = b;
        }
    }
    static void zeros(size_t size, T* output) {
        set(0, size, output);
    }
    static void zeros(size_t size, size_t* mask, T* output) {
        set(0, size, mask, output);
    }
    static void ones(size_t size, T* output) {
        set(1, size, output);
    }
    static void ones(size_t size, size_t* mask, T* output) {
        set(1, size, mask, output);
    }
    static void find(const T* a, T fn(T, T), size_t size, T output[2]) 
    {
        // size find
        for (size_t i=0; i < size; ++i) {
            if (fn(a[i], output[0])) {
                output[0] = a[i];
                output[1] = T(i);
            }
        }
    }
    static void find(const T* a, T fn(T, T), size_t size, size_t* mask, T output[2]) 
    {
        // mask find
        for (size_t i=0; i < size; ++i) {
            if (fn(a[mask[i]], output[0])) {
                output[0] = a[mask[i]];
                output[1] = T(mask[i]);
            }
        }
    }
    // max and min
    static T max(const T* a, size_t size) {
        T output[2] = {-INFINITY, -1};
        find(a, gt, size, output);
        return output[0];
    }
    static T max(const T* a, size_t size, size_t* mask) {
        T output[2] = {-INFINITY, -1};
        find(a, gt, size, mask, output);
        return output[0];
    }
    static T argmax(const T* a, size_t size) {
        T output[2] = {-INFINITY, -1};
        find(a, gt, size, output);
        return output[1];
    }
    static T argmax(const T* a, size_t size, size_t* mask) {
        T output[2] = {-INFINITY, -1};
        find(a, gt, size, mask, output);
        return output[1];
    }
    static T min(const T* a, size_t size) {
        T output[2] = {INFINITY, -1};
        find(a, lt, size, output);
        return output[0];
    }
    static T min(const T* a, size_t size, size_t* mask) {
        T output[2] = {INFINITY, -1};
        find(a, lt, size, mask, output);
        return output[0];
    }
    static T argmin(const T* a, size_t size) {
        T output[2] = {INFINITY, -1};
        find(a, lt, size, output);
        return output[1];
    }
    static T argmin(const T* a, size_t size, size_t* mask) {
        T output[2] = {INFINITY, -1};
        find(a, lt, size, mask, output);
        return output[1];
    }
    static void set_max(const T* a, size_t size, T* output) {
        T o[2] = {-INFINITY, -1};
        find(a, gt, size, o);
        if ((o[1] > -1) & (output[size_t(o[1])] != o[0]))
            output[size_t(o[1])] = o[0];
    }
    static void set_max(const T* a, size_t size, size_t* mask, T* output) {
        T o[2] = {-INFINITY, -1};
        find(a, gt, size, mask, o);
        if ((o[1] > -1) & (output[size_t(o[1])] != o[0]))
            output[size_t(o[1])] = o[0];
    }
    // 2x2 max pooling? convert between indices?
    // other pooling functions using mask_fn?
};
    