#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#if defined __has_include
#  if __has_include (<omp.h>)
#    include <omp.h>
#  endif
#endif
#include "pool.cpp"

template <typename T, typename fn>
static PyObject* rf_pool(PyArrayObject* array, PyArrayObject* mask, fn pool_fn)
{
    // get ndim, dims, type from array
    int ndim_a = PyArray_NDIM(array);
    npy_intp* dims_a = PyArray_DIMS(array);
    int type_a = PyArray_TYPE(array);
    
    // init mask data
    size_t size_m = PyArray_SIZE(mask);
    npy_intp* dims_m = PyArray_DIMS(mask);
    size_t mask_size = size_m / dims_m[0];
    
    // init data as zeros
    PyArrayObject* output = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, type_a, 0);
    
    // loop through batch*channels
    #pragma omp parallel for collapse(2)
    for (size_t i=0; i < size_t(dims_a[0]); ++i) {
        // loop through mask
        for (size_t j=0; j < size_t(dims_m[0]); ++j) {
            // call rf pooling function
            pool_fn((T*) PyArray_GETPTR1(array, i), mask_size,
                    (T*) PyArray_GETPTR1(mask, j), (T*) PyArray_GETPTR1(output, i));
        }
    }
    
    return (PyObject*) output;
}

template <typename T, typename fn>
static PyObject* kernel_pool(PyArrayObject* array, size_t* kernel, size_t* img_shape,
                             size_t* stride, fn pool_fn)
{
    // get ndim, dims, type from array
    int ndim_a = PyArray_NDIM(array);
    npy_intp* dims_a = PyArray_DIMS(array);
    int type_a = PyArray_TYPE(array);
    size_t size_a = PyArray_SIZE(array);
    
    // init data as zeros
    int cnt = 0;
    for (int i=0; i < ndim_a; ++i) {
        if (i >= ndim_a - 2) {
            dims_a[i] = (dims_a[i] - kernel[cnt]) / stride[cnt] + 1;
            ++cnt;
        }
    }
    PyArrayObject* output = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, type_a, 0);
    
    // call kernel pooling function
    pool_fn((T*) PyArray_DATA(array), kernel, img_shape, stride,
            size_a, (T*) PyArray_DATA(output));
    
    return (PyObject*) output;
}

template <typename T>
static void parse_list_args(PyObject* list_arg, size_t size, T* output) {
    if (list_arg == NULL) {
        return;
    }
    PyObject* iter = PyObject_GetIter(list_arg);
    for (size_t i=0; i < size; ++i) {
        PyObject *next = PyIter_Next(iter);
        output[i] = PyFloat_AsDouble(next);
        Py_DECREF(next);
    }
    Py_DECREF(iter);
}

template <typename T, typename rf_fn, typename kernel_fn>
static PyObject* apply(PyObject* self, PyObject* args, PyObject* kwargs,
                       rf_fn rf_pool_fn, kernel_fn kernel_pool_fn,
                       bool reverse = false)
{
    PyArrayObject* array = NULL;
    PyArrayObject* mask = NULL;
    PyObject* kernel_obj = NULL;
    PyObject* img_shape_obj = NULL;
    PyObject* stride_obj = NULL;

    static char const* kwlist[] = {"array", "mask", "kernel", "img_shape", "stride", NULL};
    // parse inputs
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O!OOO", 
                                     const_cast<char**>(kwlist), 
                                     &PyArray_Type, &array,
                                     &PyArray_Type, &mask,
                                     &kernel_obj, &img_shape_obj, &stride_obj))
        return NULL;
    
    // get list vars
    size_t kernel[2];
    parse_list_args<size_t>(kernel_obj, 2, kernel);
    size_t img_shape[2]; 
    parse_list_args<size_t>(img_shape_obj, 2, img_shape);
    size_t stride[2];
    parse_list_args<size_t>(stride_obj, 2, stride);
    if ((stride_obj == NULL) & (kernel_obj != NULL)) {
        stride[0] = kernel[0];
        stride[1] = kernel[1];
    }
    
    // apply pooling operations
    PyObject* output = (PyObject*) array;
    for (size_t i=0; i < 2; ++i) {
        if ((mask != NULL) & (reverse == bool(i))) {
            // rf_pool
            output = rf_pool<T, rf_fn>((PyArrayObject*) output, mask, rf_pool_fn);
        } else if ((kernel_obj != NULL) & (reverse != bool(i))) {
            // kernel pool
            output = kernel_pool<T, kernel_fn>((PyArrayObject*) output, kernel, img_shape,
                                               stride, kernel_pool_fn);
        }
    }
    return output;
}

template <typename T, typename rf_fn, typename kernel_fn>
static PyObject* max_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // init pool class
    pool<T> tpool;
    
    // apply pooling functions
    return apply<T, rf_fn, kernel_fn>(self, args, kwargs, tpool.rf_max_pool, tpool.kernel_max_pool);
}

template <typename T, typename rf_fn, typename kernel_fn>
static PyObject* avg_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // init pool class
    pool<T> pool;
    
    // apply pooling functions
    return apply<T, rf_fn, kernel_fn>(self, args, kwargs, pool.rf_avg_pool, pool.kernel_avg_pool);
}

template <typename T, typename rf_fn, typename kernel_fn>
static PyObject* sum_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // init pool class
    pool<T> pool;
    
    // apply pooling functions
    return apply<T, rf_fn, kernel_fn>(self, args, kwargs, pool.rf_sum_pool, pool.kernel_sum_pool);
}

typedef void (rf_fn)(const float*, size_t, const float*, float*);
typedef void (kernel_fn)(const float*, size_t*, size_t*, size_t*, size_t, float*);
static PyMethodDef pool_methods[] = {
    {"max_pool", 
     (PyCFunction) max_pool<float, rf_fn, kernel_fn>,
     METH_VARARGS | METH_KEYWORDS, 
     "max_pool(array, mask=None, kernel=None, img_shape=None, stride=None)\n\n \
     Ouput max across each receptive field (if mask given) at max index,\n \
     followed by max across kernel (if given).\n\n \
     Parameters\n \
     ----------\n \
     array : numpy.ndarray\n \
         input array with shape (batch*channel, img_height, img_width)\n \
     mask : numpy.ndarray or None\n \
         receptive field pooling mask with shape(n_RFs, img_height, img_width)\n \
     kernel : tuple or None\n \
         pooling kernel to apply for kernel pooling (e.g., 2x2 MaxPool)\n \
     img_shape : tuple or None\n \
         image shape used for kernel pooling\n \
     stride : tuple or None (optional)\n \
         stride used for kernel pooling [default : kernel]\n\n \
     Returns\n \
     -------\n \
     output : numpy.ndarray\n \
         output array with shape (batch*channel, img_height, img_width)\n \
         with the maximum value within each receptive field maintained and\n \
         all other values set to zero (if mask != None). If kernel is given\n \
         kernel pooling is subsequently applied to the output (or input)."},
    {"avg_pool", 
     (PyCFunction) avg_pool<float, rf_fn, kernel_fn>,
     METH_VARARGS | METH_KEYWORDS, 
     "avg_pool(array, mask=None, kernel=None, img_shape=None, stride=None)\n\n \
     Ouput average across each receptive field (if mask given) at max index,\n \
     followed by average across kernel (if given).\n\n \
     Parameters\n \
     ----------\n \
     array : numpy.ndarray\n \
         input array with shape (batch*channel, img_height, img_width)\n \
     mask : numpy.ndarray or None\n \
         receptive field pooling mask with shape(n_RFs, img_height, img_width)\n \
     kernel : tuple or None\n \
         pooling kernel to apply for kernel pooling (e.g., 2x2 MaxPool)\n \
     img_shape : tuple or None\n \
         image shape used for kernel pooling\n \
     stride : tuple or None (optional)\n \
         stride used for kernel pooling [default : kernel]\n\n \
     Returns\n \
     -------\n \
     output : numpy.ndarray\n \
         output array with shape (batch*channel, img_height, img_width)\n \
         with the maximum value within each receptive field maintained and\n \
         all other values set to zero (if mask != None). If kernel is given\n \
         kernel pooling is subsequently applied to the output (or input)."},
    {"sum_pool", 
     (PyCFunction) sum_pool<float, rf_fn, kernel_fn>,
     METH_VARARGS | METH_KEYWORDS, 
     "sum_pool(array, mask=None, kernel=None, img_shape=None, stride=None)\n\n \
     Ouput sum across each receptive field (if mask given) at max index,\n \
     followed by sum across kernel (if given).\n\n \
     Parameters\n \
     ----------\n \
     array : numpy.ndarray\n \
         input array with shape (batch*channel, img_height, img_width)\n \
     mask : numpy.ndarray or None\n \
         receptive field pooling mask with shape(n_RFs, img_height, img_width)\n \
     kernel : tuple or None\n \
         pooling kernel to apply for kernel pooling (e.g., 2x2 MaxPool)\n \
     img_shape : tuple or None\n \
         image shape used for kernel pooling\n \
     stride : tuple or None (optional)\n \
         stride used for kernel pooling [default : kernel]\n\n \
     Returns\n \
     -------\n \
     output : numpy.ndarray\n \
         output array with shape (batch*channel, img_height, img_width)\n \
         with the maximum value within each receptive field maintained and\n \
         all other values set to zero (if mask != None). If kernel is given\n \
         kernel pooling is subsequently applied to the output (or input)."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef pool_definition = {
    PyModuleDef_HEAD_INIT,
    "pool",
    "Receptive Field Pooling Functions",
    -1,
    pool_methods
};

PyMODINIT_FUNC PyInit_pool(void) {
    import_array();
    return PyModule_Create(&pool_definition);
};