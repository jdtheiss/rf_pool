#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#if defined __has_include
#  if __has_include (<omp.h>)
#    include <omp.h>
#  endif
#endif
#include "c_ops.h"

template <typename T>
class pool {
public:
    c_ops<T> ops;
    // max pool operation, set output to max value in mask at index
    void max_pool(T* array, size_t mask_size, T* mask, T* output) {
        size_t* mask_index = (size_t*) malloc(mask_size * sizeof(size_t));
        size_t index_size = ops.where(mask, 0, ops.gt, mask_size, mask_index);
        ops.set_max(array, index_size, mask_index, output);
        free(mask_index);
    }
};

template <typename T>
static PyObject* max_pool(PyObject* self, PyObject* args)
{
    PyArrayObject* array = NULL;
    PyArrayObject* mask = NULL;

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &array, &PyArray_Type, &mask))
        return NULL;
    
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
    
    // init pool class
    pool<T> pool;
    
    // loop through batch*channels
    #pragma omp parallel for collapse(2)
    for (size_t i=0; i < size_t(dims_a[0]); ++i) {
        // loop through mask
        for (size_t j=0; j < size_t(dims_m[0]); ++j) {
            pool.max_pool((T*) PyArray_GETPTR1(array, i), mask_size,
                          (T*) PyArray_GETPTR1(mask, j), (T*) PyArray_GETPTR1(output, i));
        }
    }
    
    return (PyObject*) output;
}

static PyMethodDef pool_methods[] = {
    {"max_pool", max_pool<float>, METH_VARARGS, 
     "max_pool(array, mask)\n \
     Parameters\n \
     ----------\n \
     array : numpy.ndarray\n \
         input array with shape (batch*channel, img_height, img_width)\n \
     mask : numpy.ndarray\n \
         receptive field pooling mask with shape(n_RFs, img_height, img_width)\n \
     Returns\n \
     -------\n \
     output : numpy.ndarray\n \
         output array with shape (batch*channel, img_height, img_width)\n \
         with the maximum value within each receptive field maintained and\n \
         all other values set to zero."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef pool_definition = {
    PyModuleDef_HEAD_INIT,
    "pool",
    "pool module containing max_pool() function",
    -1,
    pool_methods,
};

PyMODINIT_FUNC PyInit_pool(void) {
    
    import_array();
    return PyModule_Create(&pool_definition);
};    
