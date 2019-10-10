#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#if defined __has_include
#  if __has_include (<omp.h>)
#    include <omp.h>
#  endif
#endif
#include "pool.cpp"

static pool<float> P;

template<typename T, typename fn>
static PyObject* rf_pool(PyArrayObject* array, PyArrayObject* mask, fn pool_fn, bool retain_shape)
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
    // if retain_shape set output to (batch*ch, n_rfs, img_h, img_w)
    PyArrayObject* output;
    if (retain_shape) {
        ndim_a = 4;
        npy_intp new_dims[4];
        int cnt = 0;
        for (int i=0; i < ndim_a; ++i) {
            if ((i == 1) & (retain_shape)) {
                new_dims[i] = dims_m[0];
            } else {
                new_dims[i] = dims_a[cnt];
                ++cnt;
            }
        }
        output = (PyArrayObject*) PyArray_ZEROS(ndim_a, new_dims, type_a, 0);
    } else {
        output = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, type_a, 0);
    }
    
    // loop through batch*channels
    #pragma omp parallel for collapse(2)
    for (size_t i=0; i < size_t(dims_a[0]); ++i) {
        // loop through mask
        for (size_t j=0; j < size_t(dims_m[0]); ++j) {
            if (retain_shape) {
                // call rf pooling function while retaining mask shape
                pool_fn((T*) PyArray_GETPTR1(array, i), mask_size,
                        (T*) PyArray_GETPTR1(mask, j), (T*) PyArray_GETPTR2(output, i, j));
            } else {
                // call rf pooling function
                pool_fn((T*) PyArray_GETPTR1(array, i), mask_size,
                        (T*) PyArray_GETPTR1(mask, j), (T*) PyArray_GETPTR1(output, i));
            }
        }
    }
    
    return (PyObject*) output;
}

template<typename T, typename fn>
static PyObject* kernel_pool(PyArrayObject* array, size_t* kernel, size_t* img_shape,
                             size_t* stride, fn pool_fn, bool subsample = true)
{
    // get ndim, dims, type from array
    int ndim_a = PyArray_NDIM(array);
    npy_intp* dims_a = PyArray_DIMS(array);
    int type_a = PyArray_TYPE(array);
    size_t size_a = PyArray_SIZE(array);
    
    // init data as zeros with subsampled shape
    if (subsample) {
        int cnt = 0;
        for (int i=0; i < ndim_a; ++i) {
            if (i >= ndim_a - 2) {
                dims_a[i] = (dims_a[i] - kernel[cnt]) / stride[cnt] + 1;
                ++cnt;
            }
        }
    }
    PyArrayObject* output = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, type_a, 0);
    
    // call kernel pooling function
    pool_fn((T*) PyArray_DATA(array), kernel, img_shape, stride, size_a, 
            (T*) PyArray_DATA(output));
    
    return (PyObject*) output;
}

template<typename T>
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

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* apply(PyObject* self, PyObject* args, PyObject* kwargs,
                       rf_fn rf_pool_fn, kernel_fn kernel_pool_fn,
                       bool kernel_first = false, bool subsample = true)
{
    PyArrayObject* array = NULL;
    PyArrayObject* mask = NULL;
    PyObject* kernel_obj = NULL;
    PyObject* img_shape_obj = NULL;
    PyObject* stride_obj = NULL;
    bool* retain_shape = (bool*) false;

    static char const* kwlist[] = {"array", "mask", "kernel", "img_shape",
                                   "stride", "retain_shape", NULL};
    // parse inputs
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O!OOOp", 
                                     const_cast<char**>(kwlist), 
                                     &PyArray_Type, &array,
                                     &PyArray_Type, &mask,
                                     &kernel_obj, &img_shape_obj, &stride_obj,
                                     &retain_shape))
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
        if ((mask != NULL) & (kernel_first == bool(i))) {
            // rf_pool
            output = rf_pool<T, rf_fn>((PyArrayObject*) output, mask, rf_pool_fn, retain_shape);
        } else if ((kernel_obj != NULL) & (kernel_first != bool(i))) {
            // kernel pool
            output = kernel_pool<T, kernel_fn>((PyArrayObject*) output, kernel, img_shape,
                                               stride, kernel_pool_fn, subsample);
        }
    }
    return output;
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* max_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{   
    // apply pooling functions
    return apply<T, rf_fn, kernel_fn>(self, args, kwargs, P.rf_max_pool, P.kernel_max_pool);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* avg_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // apply pooling functions
    return apply<T, rf_fn, kernel_fn>(self, args, kwargs, P.rf_avg_pool, P.kernel_avg_pool);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* sum_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // apply pooling functions
    return apply<T, rf_fn, kernel_fn>(self, args, kwargs, P.rf_sum_pool, P.kernel_sum_pool);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* probmax(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // apply pooling functions
    return apply<T, rf_fn, kernel_fn>(self, args, kwargs, P.rf_probmax_pool, 
                                      P.kernel_probmax, true, false);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* probmax_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // apply pooling functions
    return apply<T, rf_fn, kernel_fn>(self, args, kwargs, P.rf_probmax_pool, 
                                      P.kernel_probmax_pool, true);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* stochastic_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{   
    // apply pooling functions
    return apply<T, rf_fn, kernel_fn>(self, args, kwargs, P.rf_stochastic_pool,
                                      P.kernel_stochastic_pool);
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
         output array with shape (batch*channel, img_height/kernel[0], img_width/kernel[1])\n \
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
         output array with shape (batch*channel, img_height/kernel[0], img_width/kernel[1])\n \
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
         output array with shape (batch*channel, img_height/kernel[0], img_width/kernel[1])\n \
         with the maximum value within each receptive field maintained and\n \
         all other values set to zero (if mask != None). If kernel is given\n \
         kernel pooling is subsequently applied to the output (or input)."},
    {"probmax", 
     (PyCFunction) probmax<float, rf_fn, kernel_fn>,
     METH_VARARGS | METH_KEYWORDS, 
     "probmax(array, mask=None, kernel=None, img_shape=None, stride=None)\n\n \
     Ouput (1 - prob(all pixels off)) across kernel (if given) at multinomial index,\n \
     followed by (1 - prob(all pixels off)) across each receptive field (if mask given)\n \
     at multinomial index (retains input image shape).\n\n \
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
         kernel operation is applied to the input followed by rf operation.\n\n \
     References\n \
     ----------\n \
     Lee, H., Grosse, R., Ranganath, R., & Ng, A. Y. (2009, June). Convolutional\n \
     deep belief networks for scalable unsupervised learning of hierarchical\n \
     representations. In Proceedings of the 26th annual international conference\n \
     on machine learning (pp. 609-616). ACM."},
    {"probmax_pool", 
     (PyCFunction) probmax_pool<float, rf_fn, kernel_fn>,
     METH_VARARGS | METH_KEYWORDS, 
     "probmax_pool(array, mask=None, kernel=None, img_shape=None, stride=None)\n\n \
     Ouput (1 - prob(all pixels off)) across kernel (if given),\n \
     followed by (1 - prob(all pixels off)) across each receptive field (if mask given)\n \
     at multinomial index.\n\n \
     Parameters\n \
     ----------\n \
     array : numpy.ndarray\n \
         input array with shape (batch*channel, img_height, img_width)\n \
     mask : numpy.ndarray or None\n \
         receptive field pooling mask with shape(n_RFs, img_height/kernel[0], img_width/kernel[1])\n \
     kernel : tuple or None\n \
         pooling kernel to apply for kernel pooling (e.g., 2x2 MaxPool)\n \
     img_shape : tuple or None\n \
         image shape used for kernel pooling\n \
     stride : tuple or None (optional)\n \
         stride used for kernel pooling [default : kernel]\n\n \
     Returns\n \
     -------\n \
     output : numpy.ndarray\n \
         output array with shape (batch*channel, img_height/kernel[0], img_width/kernel[1])\n \
         with the maximum value within each receptive field maintained and\n \
         all other values set to zero (if mask != None). If kernel is given\n \
         kernel pooling is applied to the input followed by rf pooling.\n\n \
     References\n \
     ----------\n \
     Lee, H., Grosse, R., Ranganath, R., & Ng, A. Y. (2009, June). Convolutional\n \
     deep belief networks for scalable unsupervised learning of hierarchical\n \
     representations. In Proceedings of the 26th annual international conference\n \
     on machine learning (pp. 609-616). ACM."},
    {"stochastic_pool", 
     (PyCFunction) stochastic_pool<float, rf_fn, kernel_fn>,
     METH_VARARGS | METH_KEYWORDS, 
     "stochastic_pool(array, mask=None, kernel=None, img_shape=None, stride=None)\n\n \
     Ouput max across each receptive field (if mask given) at multinomial index,\n \
     followed by max across kernel (if given) at multinomial index.\n\n \
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
         output array with shape (batch*channel, img_height/kernel[0], img_width/kernel[1])\n \
         with the maximum value within each receptive field maintained and\n \
         all other values set to zero (if mask != None). If kernel is given\n \
         kernel pooling is subsequently applied to the output (or input).\n\n \
     References\n \
     ----------\n \
     Zeiler, M. D., & Fergus, R. (2013). Stochastic pooling for regularization\n \
     of deep convolutional neural networks. arXiv preprint arXiv:1301.3557."},
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