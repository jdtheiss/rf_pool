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

template<typename T>
static void parse_list_args(PyObject* list_arg, size_t size, T* output) {
    if (list_arg == Py_None) {
        return;
    }
    PyObject* iter;
    int len_arg;
    if (PyObject_HasAttrString(list_arg, "__iter__")) {
        iter = PyObject_GetIter(list_arg);
        len_arg = int(PyObject_Length(list_arg));
    } else {
        iter = NULL;
        len_arg = 0;
    }
    for (int i=0; i < int(size); ++i) {
        if (len_arg > i) {
            PyObject *next = PyIter_Next(iter);
            output[i] = PyFloat_AsDouble(next);
            Py_DECREF(next);
        } else if (len_arg > 0) {
            output[i] = output[i-1];
        } else {
            output[i] = PyFloat_AsDouble(list_arg);
        }
    }
    if (iter != NULL) {
        Py_DECREF(iter);
    }
}

template<typename T, typename fn>
static PyObject* rf_pool(PyObject* args, fn pool_fn)
{
    // get inputs
    size_t len_args = PyList_Size(args);
    PyArrayObject* array = (PyArrayObject*) PyList_GetItem(args, 0);
    PyArrayObject* mask_indices = (PyArrayObject*) PyArray_FROM_O(PyList_GetItem(args, 1));
    bool retain_shape = PyObject_IsTrue(PyList_GetItem(args, len_args-1));
    
    // get ndim, dims, type from array
    int ndim_a = PyArray_NDIM(array);
    npy_intp* dims_a = PyArray_DIMS(array);
    int type_a = PyArray_TYPE(array);
    
    // init mask_indices data
    size_t size_m = PyArray_SIZE(mask_indices);
    npy_intp* dims_m = PyArray_DIMS(mask_indices);
    size_t mask_size = size_m / dims_m[0];
    
    // init data as zeros 
    // if retain_shape set output to (batch*ch, n_rfs, img_h, img_w)
    PyArrayObject* output;
    PyArrayObject* indices;
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
        indices = (PyArrayObject*) PyArray_ZEROS(ndim_a, new_dims, NPY_LONG, 0);
    } else {
        output = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, type_a, 0);
        indices = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, NPY_LONG, 0);
    }
    
    // loop through batch*channels
    #pragma omp parallel for collapse(2)
    for (size_t i=0; i < size_t(dims_a[0]); ++i) {
        // loop through mask_indices
        for (size_t j=0; j < size_t(dims_m[0]); ++j) {
            if (retain_shape) {
                // call rf pooling function while retaining mask_indices shape
                pool_fn((T*) PyArray_GETPTR1(array, i), mask_size, (T*) PyArray_GETPTR1(mask_indices, j),
                        (T*) PyArray_GETPTR2(output, i, j), (size_t*) PyArray_GETPTR2(indices, i, j));
            } else {
                // call rf pooling function
                pool_fn((T*) PyArray_GETPTR1(array, i), mask_size, (T*) PyArray_GETPTR1(mask_indices, j),
                        (T*) PyArray_GETPTR1(output, i), (size_t*) PyArray_GETPTR1(indices, i));
            }
        }
    }
    Py_DECREF(mask_indices);
    return Py_BuildValue("(OO)", (PyObject*) output, (PyObject*) indices);
}

template<typename T, typename fn>
static PyObject* kernel_pool(PyObject* args, fn pool_fn, bool subsample = true)
{
    // get array
    PyArrayObject* array = (PyArrayObject*) PyList_GetItem(args, 0);
    
    // get list vars
    size_t kernel[2];
    PyObject* kernel_obj = PyList_GetItem(args, 2);
    parse_list_args<size_t>(kernel_obj, 2, kernel);
    size_t stride[2];
    PyObject* stride_obj = PyList_GetItem(args, 3);
    parse_list_args<size_t>(stride_obj, 2, stride);
    if ((stride_obj == Py_None) & (kernel_obj != Py_None)) {
        stride[0] = kernel[0];
        stride[1] = kernel[1];
    }
    
    // get ndim, dims, type from array
    int ndim_a = PyArray_NDIM(array);
    npy_intp* dims_a = PyArray_DIMS(array);
    int type_a = PyArray_TYPE(array);
    
    // set img_shape
    size_t img_shape[] = {size_t(dims_a[ndim_a - 2]), size_t(dims_a[ndim_a - 1])};
    size_t img_size = img_shape[0] * img_shape[1];
    
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
    PyArrayObject* indices = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, NPY_LONG, 0);
    
    // loop through batch*channels
    #pragma omp parallel for 
    for (size_t i=0; i < size_t(dims_a[0]); ++i) {
        pool_fn((T*) PyArray_GETPTR1(array, i), kernel, img_shape, stride, img_size, 
                (T*) PyArray_GETPTR1(output, i), (size_t*) PyArray_GETPTR1(indices, i));
    }
    
    return Py_BuildValue("(OO)", (PyObject*) output, (PyObject*) indices);
}

template<typename T, typename fn>
static PyObject* kernel_unpool(PyObject* args, fn unpool_fn)
{
    // get array
    PyArrayObject* array = (PyArrayObject*) PyList_GetItem(args, 0);
    
    // get list vars
    size_t kernel[2];
    PyObject* kernel_obj = PyList_GetItem(args, 2);
    parse_list_args<size_t>(kernel_obj, 2, kernel);
    size_t stride[2];
    PyObject* stride_obj = PyList_GetItem(args, 3);
    parse_list_args<size_t>(stride_obj, 2, stride);
    if ((stride_obj == Py_None) & (kernel_obj != Py_None)) {
        stride[0] = kernel[0];
        stride[1] = kernel[1];
    }
    
    // get ndim, dims, type from array
    int ndim_a = PyArray_NDIM(array);
    npy_intp* dims_a = PyArray_DIMS(array);
    int type_a = PyArray_TYPE(array);
    
    // init data as zeros with un-subsampled shape
    int cnt = 0;
    for (int i=0; i < ndim_a; ++i) {
        if (i >= ndim_a - 2) {
            dims_a[i] = (dims_a[i] - 1) * stride[cnt] + kernel[cnt];
            ++cnt;
        }
    }
    PyArrayObject* output = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, type_a, 0);
    
    // get mask
    PyArrayObject* mask;
    if (PyList_GetItem(args, 1) != Py_None) {
        mask = (PyArrayObject*) PyArray_FROM_OT(PyList_GetItem(args, 1), type_a);
    } else {
        mask = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, type_a, 0);
        PyArray_FillWithScalar(mask, PyFloat_FromDouble(1));
    }
    
    // set img_shape
    size_t img_shape[] = {size_t(dims_a[ndim_a - 2]), size_t(dims_a[ndim_a - 1])};
    
    // call kernel pooling function
    unpool_fn((T*) PyArray_DATA(array), kernel, img_shape, stride, PyArray_SIZE(output), 
              (T*) PyArray_DATA(mask), (T*) PyArray_DATA(output));
    Py_DECREF(mask);
    return (PyObject*) output;
}

static PyObject* parse_args(PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyArrayObject* array = NULL;
    PyObject* mask_obj = Py_None;
    PyObject* kernel_obj = Py_None;
    PyObject* stride_obj = Py_None;
    PyObject* retain_shape = Py_False;

    static char const* kwlist[] = {"array", "mask_indices", "kernel_size",
                                   "stride", "retain_shape", NULL};
    // parse inputs
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|OOOO", 
                                     const_cast<char**>(kwlist), 
                                     &PyArray_Type, &array, &mask_obj,
                                     &kernel_obj, &stride_obj,
                                     &retain_shape))
        return NULL;
    
    // return list of input args/kwargs
    return Py_BuildValue("[OOOOO]", (PyObject*) array, mask_obj, kernel_obj, stride_obj, retain_shape);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* max_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{   
    // apply pooling functions
    PyObject* inputs = parse_args(self, args, kwargs);
    PyObject* output_tuple = Py_None;
    PyObject* index_mask = Py_None;
    if (PyList_GetItem(inputs, 1) != Py_None) {
        output_tuple = rf_pool<T, rf_fn>(inputs, P.rf_max_pool);
        index_mask = PyTuple_GetItem(output_tuple, 1);
        PyList_SetItem(inputs, 0, PyTuple_GetItem(output_tuple, 0));
    }
    PyObject* index_kernel = Py_None;
    if (PyList_GetItem(inputs, 2) != Py_None) {
        output_tuple = kernel_pool<T, kernel_fn>(inputs, P.kernel_max_pool);
        index_kernel = PyTuple_GetItem(output_tuple, 1);
    }
    PyObject* output = PyTuple_GetItem(output_tuple, 0);
    Py_DECREF(output_tuple);
    return Py_BuildValue("(OOO)", output, index_mask, index_kernel);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* probmax(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // apply pooling functions
    PyObject* inputs = parse_args(self, args, kwargs);
    PyObject* output_tuple = Py_None;
    PyObject* index_mask = Py_None;
    PyObject* index_kernel = Py_None;
    if (PyList_GetItem(inputs, 2) != Py_None) {
        output_tuple = kernel_pool<T, kernel_fn>(inputs, P.kernel_probmax_pool, false);
        index_kernel = PyTuple_GetItem(output_tuple, 1);
    }
    if (PyList_GetItem(inputs, 1) != Py_None) {
        output_tuple = rf_pool<T, rf_fn>(inputs, P.rf_probmax_pool);
        index_mask = PyTuple_GetItem(output_tuple, 1);
        PyList_SetItem(inputs, 0, PyTuple_GetItem(output_tuple, 0));
    }
    PyObject* output = PyTuple_GetItem(output_tuple, 0);
    Py_DECREF(output_tuple);
    return Py_BuildValue("(OOO)", output, index_mask, index_kernel);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* probmax_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // apply pooling functions
    PyObject* inputs = parse_args(self, args, kwargs);
    PyObject* output_tuple = Py_None;
    PyObject* index_mask = Py_None;
    PyObject* index_kernel = Py_None;
    if (PyList_GetItem(inputs, 2) != Py_None) {
        output_tuple = kernel_pool<T, kernel_fn>(inputs, P.kernel_probmax_pool);
        index_kernel = PyTuple_GetItem(output_tuple, 1);
    }
    if (PyList_GetItem(inputs, 1) != Py_None) {
        output_tuple = rf_pool<T, rf_fn>(inputs, P.rf_probmax_pool);
        index_mask = PyTuple_GetItem(output_tuple, 1);
        PyList_SetItem(inputs, 0, PyTuple_GetItem(output_tuple, 0));
    }
    PyObject* output = PyTuple_GetItem(output_tuple, 0);
    Py_DECREF(output_tuple);
    return Py_BuildValue("(OOO)", output, index_mask, index_kernel);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* stochastic_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{   
    // apply pooling functions
    PyObject* inputs = parse_args(self, args, kwargs);
    PyObject* output_tuple = Py_None;
    PyObject* index_mask = Py_None;
    if (PyList_GetItem(inputs, 1) != Py_None) {
        output_tuple = rf_pool<T, rf_fn>(inputs, P.rf_stochastic_pool);
        index_mask = PyTuple_GetItem(output_tuple, 1);
        PyList_SetItem(inputs, 0, PyTuple_GetItem(output_tuple, 0));
    }
    PyObject* index_kernel = Py_None;
    if (PyList_GetItem(inputs, 2) != Py_None) {
        output_tuple = kernel_pool<T, kernel_fn>(inputs, P.kernel_stochastic_pool);
        index_kernel = PyTuple_GetItem(output_tuple, 1);
    }
    PyObject* output = PyTuple_GetItem(output_tuple, 0);
    Py_DECREF(output_tuple);
    return Py_BuildValue("(OOO)", output, index_mask, index_kernel);
}

template<typename T, typename grad_fn>
static PyObject* unpool(PyObject* self, PyObject* args, PyObject* kwargs)
{   
    // apply unpooling functions
    PyObject* input_tuple = parse_args(self, args, kwargs);
    PyObject* output = kernel_unpool<T, grad_fn>(input_tuple, P.kernel_unpool);
    Py_DECREF(input_tuple);
    return Py_BuildValue("(O)", output);
}

typedef void (rf_fn)(const float*, size_t, const float*, float*, size_t*);
typedef void (kernel_fn)(const float*, size_t*, size_t*, size_t*, size_t, float*, size_t*);
typedef void (grad_fn)(const float*, size_t*, size_t*, size_t*, size_t, const float*, float*);
static PyMethodDef pool_methods[] = {
    {"max_pool", 
     (PyCFunction) max_pool<float, rf_fn, kernel_fn>,
     METH_VARARGS | METH_KEYWORDS, 
     "max_pool(array, mask_indices=None, kernel=None, img_shape=None, stride=None)\n\n \
     Ouput max across each receptive field (if mask_indices given) at max index,\n \
     followed by max across kernel (if given).\n\n \
     Parameters\n \
     ----------\n \
     array : numpy.ndarray\n \
         input array with shape (batch*channel, img_height, img_width)\n \
     mask_indices : numpy.ndarray or None\n \
         receptive field pooling indices with shape(n_RFs, img_height * img_width)\n \
     kernel_size : tuple or int, optional\n \
         pooling kernel to apply for kernel pooling (e.g., 2x2 MaxPool)\n \
     img_shape : tuple or int, optional\n \
         image shape used for kernel pooling\n \
     stride : tuple or int, optional\n \
         stride used for kernel pooling [default : kernel]\n\n \
     Returns\n \
     -------\n \
     output : numpy.ndarray\n \
         output array with shape (batch*channel, img_height/kernel[0], img_width/kernel[1])\n \
         with the maximum value within each receptive field maintained and\n \
         all other values set to zero (if mask_indices != None). If kernel is given\n \
         kernel pooling is subsequently applied to the output (or input)."},
    {"probmax", 
     (PyCFunction) probmax<float, rf_fn, kernel_fn>,
     METH_VARARGS | METH_KEYWORDS, 
     "probmax(array, mask_indices=None, kernel=None, img_shape=None, stride=None)\n\n \
     Ouput (1 - prob(all pixels off)) across kernel (if given) at multinomial index,\n \
     followed by (1 - prob(all pixels off)) across each receptive field (if mask_indices given)\n \
     at multinomial index (retains input image shape).\n\n \
     Parameters\n \
     ----------\n \
     array : numpy.ndarray\n \
         input array with shape (batch*channel, img_height, img_width)\n \
     mask_indices : numpy.ndarray or None\n \
         receptive field pooling indices with shape(n_RFs, img_height * img_width)\n \
     kernel_size : tuple or int, optional\n \
         pooling kernel to apply for kernel pooling (e.g., 2x2 MaxPool)\n \
     img_shape : tuple or int, optional\n \
         image shape used for kernel pooling\n \
     stride : tuple or int, optional\n \
         stride used for kernel pooling [default : kernel]\n\n \
     Returns\n \
     -------\n \
     output : numpy.ndarray\n \
         output array with shape (batch*channel, img_height, img_width)\n \
         with the maximum value within each receptive field maintained and\n \
         all other values set to zero (if mask_indices != None). If kernel is given\n \
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
     "probmax_pool(array, mask_indices=None, kernel=None, img_shape=None, stride=None)\n\n \
     Ouput (1 - prob(all pixels off)) across kernel (if given),\n \
     followed by (1 - prob(all pixels off)) across each receptive field (if mask_indices given)\n \
     at multinomial index.\n\n \
     Parameters\n \
     ----------\n \
     array : numpy.ndarray\n \
         input array with shape (batch*channel, img_height, img_width)\n \
     mask_indices : numpy.ndarray or None\n \
         receptive field pooling indices with shape(n_RFs, img_height/kernel[0] * img_width/kernel[1])\n \
     kernel_size : tuple or int, optional\n \
         pooling kernel to apply for kernel pooling (e.g., 2x2 MaxPool)\n \
     img_shape : tuple or int, optional\n \
         image shape used for kernel pooling\n \
     stride : tuple or int, optional\n \
         stride used for kernel pooling [default : kernel]\n\n \
     Returns\n \
     -------\n \
     output : numpy.ndarray\n \
         output array with shape (batch*channel, img_height/kernel[0], img_width/kernel[1])\n \
         with the maximum value within each receptive field maintained and\n \
         all other values set to zero (if mask_indices != None). If kernel is given\n \
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
     "stochastic_pool(array, mask_indices=None, kernel=None, img_shape=None, stride=None)\n\n \
     Ouput max across each receptive field (if mask_indices given) at multinomial index,\n \
     followed by max across kernel (if given) at multinomial index.\n\n \
     Parameters\n \
     ----------\n \
     array : numpy.ndarray\n \
         input array with shape (batch*channel, img_height, img_width)\n \
     mask_indices : numpy.ndarray or None\n \
         receptive field pooling indices with shape(n_RFs, img_height * img_width)\n \
     kernel_size : tuple or int, optional\n \
         pooling kernel to apply for kernel pooling (e.g., 2x2 MaxPool)\n \
     img_shape : tuple or int, optional\n \
         image shape used for kernel pooling\n \
     stride : tuple or int, optional\n \
         stride used for kernel pooling [default : kernel]\n\n \
     Returns\n \
     -------\n \
     output : numpy.ndarray\n \
         output array with shape (batch*channel, img_height/kernel[0], img_width/kernel[1])\n \
         with the maximum value within each receptive field maintained and\n \
         all other values set to zero (if mask_indices != None). If kernel is given\n \
         kernel pooling is subsequently applied to the output (or input).\n\n \
     References\n \
     ----------\n \
     Zeiler, M. D., & Fergus, R. (2013). Stochastic pooling for regularization\n \
     of deep convolutional neural networks. arXiv preprint arXiv:1301.3557."},
    {"unpool", 
     (PyCFunction) unpool<float, grad_fn>,
     METH_VARARGS | METH_KEYWORDS, 
     "unpool(array, mask=None, kernel=None, img_shape=None, stride=None)\n\n \
     Output values from array to a resized array at indexed locations in mask.\n\n \
     Parameters\n \
     ----------\n \
     array : numpy.ndarray\n \
         input array with shape (batch*channel, img_height, img_width)\n \
     mask : numpy.ndarray or None\n \
         receptive field pooling mask with shape(n_RFs, img_height, img_width)\n \
     kernel_size : tuple or int, optional\n \
         pooling kernel to apply for kernel pooling (e.g., 2x2 MaxPool)\n \
     img_shape : tuple or int, optional\n \
         image shape used for kernel pooling\n \
     stride : tuple or int, optional\n \
         stride used for kernel pooling [default : kernel]\n\n \
     Returns\n \
     -------\n \
     output : numpy.ndarray\n \
         output array with shape (batch*channel, img_height*kernel[0], img_width*kernel[1])."},
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