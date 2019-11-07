#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#if defined __has_include
#  if __has_include (<omp.h>)
#    include <omp.h>
#  endif
#endif
#include "pool.cpp"

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
    Py_XDECREF(iter);
}

static bool check_kwargs(PyObject* kwargs, const char* key) {
    PyObject* py_key = PyUnicode_FromString(key);
    bool ret = false;
    if (!PyDict_Contains(kwargs, py_key)) {
        ret = false;
    } else if (PyDict_GetItem(kwargs, py_key) != Py_None) {
        ret = true;
    }
    Py_DECREF(py_key);
    return ret;
}

template<typename T, typename fn>
static PyObject* rf_pool(PyObject* args, PyObject* kwargs, fn pool_fn)
{
    // get inputs
    PyArrayObject* array = NULL;
    PyArrayObject* mask_indices = NULL;
    bool retain_shape = false;
    array = (PyArrayObject*) PyArray_FROM_O(PyTuple_GetItem(args, 0));
    mask_indices = (PyArrayObject*) PyArray_FROM_O(PyDict_GetItemString(kwargs, "mask_indices"));
    if (check_kwargs(kwargs, "retain_shape")) {
        retain_shape = PyObject_IsTrue(PyDict_GetItemString(kwargs, "retain_shape"));
    }
    bool apply_mask = false;
    if (check_kwargs(kwargs, "apply_mask")) {
        apply_mask = PyObject_IsTrue(PyDict_GetItemString(kwargs, "apply_mask"));
    }
    
    // get ndim, dims, type from array
    int ndim_a = PyArray_NDIM(array);
    npy_intp* dims_a = PyArray_DIMS(array);
    int type_a = PyArray_TYPE(array);
    
    // init mask_indices data
    int ndim_m = PyArray_NDIM(mask_indices);
    size_t size_m = PyArray_SIZE(mask_indices);
    npy_intp* dims_m = PyArray_DIMS(mask_indices);
    size_t mask_size = size_m / dims_m[0];
    
    // init data as zeros 
    // if retain_shape set output to (batch*ch, n_rfs, img_h, img_w)
    PyArrayObject* output;
    PyArrayObject* indices;
    if (retain_shape) {
        npy_intp new_dims[3];
        new_dims[0] = dims_a[0] * dims_m[0];
        for (int i=1; i < ndim_a; ++i) {
            new_dims[i] = dims_a[i];
        }
        output = (PyArrayObject*) PyArray_ZEROS(ndim_a, new_dims, type_a, 0);
        indices = (PyArrayObject*) PyArray_ZEROS(ndim_a, new_dims, NPY_LONG, 0);
    } else {
        output = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, type_a, 0);
        indices = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, NPY_LONG, 0);
    }
    PyArray_ENABLEFLAGS(output, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(indices, NPY_ARRAY_OWNDATA);
    
    // get mask
    PyArrayObject* mask;
    if (apply_mask && check_kwargs(kwargs, "mask")) {
        mask = (PyArrayObject*) PyArray_FROM_O(PyDict_GetItemString(kwargs, "mask"));
    } else {
        mask = (PyArrayObject*) PyArray_ZEROS(ndim_m, dims_m, type_a, 0);
        apply_mask = false;
    }
    
    // loop through batch*channels
    #pragma omp parallel for collapse(2)
    for (size_t i=0; i < size_t(dims_a[0]); ++i) {
        // loop through mask_indices
        for (size_t j=0; j < size_t(dims_m[0]); ++j) {
            if (retain_shape) {
                // call rf pooling function while retaining mask_indices shape
                pool_fn((T*) PyArray_GETPTR1(array, i), (T*) PyArray_GETPTR1(mask, j),
                        mask_size, (T*) PyArray_GETPTR1(mask_indices, j),
                        (T*) PyArray_GETPTR1(output, i*dims_m[0] + j), 
                        (size_t*) PyArray_GETPTR1(indices, i*dims_m[0] + j), apply_mask);
            } else {
                // call rf pooling function
                pool_fn((T*) PyArray_GETPTR1(array, i), (T*) PyArray_GETPTR1(mask, j),
                        mask_size, (T*) PyArray_GETPTR1(mask_indices, j),
                        (T*) PyArray_GETPTR1(output, i), (size_t*) PyArray_GETPTR1(indices, i),
                        apply_mask);
            }
        }
    }
    Py_DECREF(array);
    Py_DECREF(mask_indices);
    Py_XDECREF(mask);
    return Py_BuildValue("(NN)", (PyObject*) output, (PyObject*) indices);
}

template<typename T, typename fn>
static PyObject* kernel_pool(PyObject* args, PyObject* kwargs, fn pool_fn, bool subsample = true)
{
    // get inputs
    PyArrayObject* array = NULL;
    size_t kernel[2];
    size_t stride[2];
    array = (PyArrayObject*) PyArray_FROM_O(PyTuple_GetItem(args, 0));
    parse_list_args<size_t>(PyDict_GetItemString(kwargs, "kernel_size"), 2, kernel);
    if (check_kwargs(kwargs, "stride")) {
        parse_list_args<size_t>(PyDict_GetItemString(kwargs, "stride"), 2, stride);
    } else {
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
    PyArray_ENABLEFLAGS(output, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS(indices, NPY_ARRAY_OWNDATA);
    
    // loop through batch*channels
    #pragma omp parallel for
    for (size_t i=0; i < size_t(dims_a[0]); ++i) {
        pool_fn((T*) PyArray_GETPTR1(array, i), kernel, img_shape, stride, img_size, 
                (T*) PyArray_GETPTR1(output, i), (size_t*) PyArray_GETPTR1(indices, i));
    }
    Py_DECREF(array);
    return Py_BuildValue("(NN)", (PyObject*) output, (PyObject*) indices);
}

template<typename T, typename fn>
static PyObject* kernel_unpool(PyObject* args, PyObject* kwargs, fn unpool_fn)
{
    // get inputs
    PyArrayObject* array = NULL;
    PyArrayObject* mask = NULL;
    size_t kernel[2];
    size_t stride[2];
    array = (PyArrayObject*) PyArray_FROM_O(PyTuple_GetItem(args, 0));
    parse_list_args<size_t>(PyDict_GetItemString(kwargs, "kernel_size"), 2, kernel);
    if (check_kwargs(kwargs, "stride")) {
        parse_list_args<size_t>(PyDict_GetItemString(kwargs, "stride"), 2, stride);
    } else {
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
    PyArray_ENABLEFLAGS(output, NPY_ARRAY_OWNDATA);
    
    // get mask
    if (PyDict_Contains(kwargs, PyUnicode_FromString("mask"))) {
        mask = (PyArrayObject*) PyArray_FROM_OT(PyDict_GetItemString(kwargs, "mask"), type_a);
    } else {
        mask = (PyArrayObject*) PyArray_ZEROS(ndim_a, dims_a, type_a, 0);
        PyArray_FillWithScalar(mask, PyFloat_FromDouble(1));
    }
    PyArray_ENABLEFLAGS(mask, NPY_ARRAY_OWNDATA);
    
    // set img_shape
    size_t img_shape[] = {size_t(dims_a[ndim_a - 2]), size_t(dims_a[ndim_a - 1])};
    
    // call kernel pooling function
    unpool_fn((T*) PyArray_DATA(array), kernel, img_shape, stride, PyArray_SIZE(output), 
              (T*) PyArray_DATA(mask), (T*) PyArray_DATA(output));
    Py_DECREF(array);
    Py_DECREF(mask);
    return (PyObject*) output;
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* max_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{   
    // apply rf pooling function
    PyObject* output_tuple = NULL;
    PyObject* index_mask = Py_None;
    if (check_kwargs(kwargs, "mask_indices")) {
        output_tuple = rf_pool<T, rf_fn>(args, kwargs, pool<T>::rf_max_pool);
        index_mask = PyTuple_GetItem(output_tuple, 1);
    }
    Py_INCREF(index_mask);
    // apply kernel pooling function
    PyObject* index_kernel = Py_None;
    if (check_kwargs(kwargs, "kernel_size")) {
        if (check_kwargs(kwargs, "mask_indices")) {
            PyObject* local_args = PyTuple_GetSlice(output_tuple, 0, 1);
            Py_DECREF(output_tuple);
            output_tuple = kernel_pool<T, kernel_fn>(local_args, kwargs, pool<T>::kernel_max_pool);
            Py_DECREF(local_args);
        } else {
            output_tuple = kernel_pool<T, kernel_fn>(args, kwargs, pool<T>::kernel_max_pool);
        }
        index_kernel = PyTuple_GetItem(output_tuple, 1);
    }
    Py_INCREF(index_kernel);
    // set output
    PyObject* output = PyTuple_GetItem(output_tuple, 0);
    Py_INCREF(output);
    Py_XDECREF(output_tuple);
    return Py_BuildValue("(NNN)", output, index_mask, index_kernel);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* probmax(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // apply kernel pooling function
    PyObject* output_tuple = NULL;
    PyObject* index_kernel = Py_None;
    if (check_kwargs(kwargs, "kernel_size")) {
        output_tuple = kernel_pool<T, kernel_fn>(args, kwargs, pool<T>::kernel_probmax, false);
        index_kernel = PyTuple_GetItem(output_tuple, 1);
    }
    Py_INCREF(index_kernel);
    // apply rf pooling function
    PyObject* index_mask = Py_None;
    if (check_kwargs(kwargs, "mask_indices")) {
        if (check_kwargs(kwargs, "kernel_size")) {
            PyObject* local_args = PyTuple_GetSlice(output_tuple, 0, 1);
            Py_DECREF(output_tuple);
            output_tuple = rf_pool<T, rf_fn>(local_args, kwargs, pool<T>::rf_probmax);
        } else {
            output_tuple = rf_pool<T, rf_fn>(args, kwargs, pool<T>::rf_probmax);
        }
        index_mask = PyTuple_GetItem(output_tuple, 1);
    }
    Py_INCREF(index_mask);
    // set output
    PyObject* output = PyTuple_GetItem(output_tuple, 0);
    Py_INCREF(output);
    Py_XDECREF(output_tuple);
    return Py_BuildValue("(NNN)", output, index_mask, index_kernel);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* probmax_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{
    // apply kernel pooling function
    PyObject* output_tuple = NULL;
    PyObject* index_kernel = Py_None;
    if (check_kwargs(kwargs, "kernel_size")) {
        output_tuple = kernel_pool<T, kernel_fn>(args, kwargs, pool<T>::kernel_probmax_pool);
        index_kernel = PyTuple_GetItem(output_tuple, 1);
    }
    Py_INCREF(index_kernel);
    // apply rf pooling function
    PyObject* index_mask = Py_None;
    if (check_kwargs(kwargs, "mask_indices")) {
        if (check_kwargs(kwargs, "kernel_size")) {
            PyObject* local_args = PyTuple_GetSlice(output_tuple, 0, 1);
            Py_DECREF(output_tuple);
            output_tuple = rf_pool<T, rf_fn>(local_args, kwargs, pool<T>::rf_probmax_pool);
        } else {
            output_tuple = rf_pool<T, rf_fn>(args, kwargs, pool<T>::rf_probmax_pool);
        }
        index_mask = PyTuple_GetItem(output_tuple, 1);
    }
    Py_INCREF(index_mask);
    // set output
    PyObject* output = PyTuple_GetItem(output_tuple, 0);
    Py_INCREF(output);
    Py_XDECREF(output_tuple);
    return Py_BuildValue("(NNN)", output, index_mask, index_kernel);
}

template<typename T, typename rf_fn, typename kernel_fn>
static PyObject* stochastic_pool(PyObject* self, PyObject* args, PyObject* kwargs)
{   
    // apply rf pooling function
    PyObject* output_tuple = NULL;
    PyObject* index_mask = Py_None;
    if (check_kwargs(kwargs, "mask_indices")) {
        output_tuple = rf_pool<T, rf_fn>(args, kwargs, pool<T>::rf_stochastic_pool);
        index_mask = PyTuple_GetItem(output_tuple, 1);
    }
    Py_INCREF(index_mask);
    // apply kernel pooling function
    PyObject* index_kernel = Py_None;
    if (check_kwargs(kwargs, "kernel_size")) {
        if (check_kwargs(kwargs, "mask_indices")) {
            PyObject* local_args = PyTuple_GetSlice(output_tuple, 0, 1);
            Py_DECREF(output_tuple);
            output_tuple = kernel_pool<T, kernel_fn>(local_args, kwargs, pool<T>::kernel_stochastic_pool);
            Py_DECREF(local_args);
        } else {
            output_tuple = kernel_pool<T, kernel_fn>(args, kwargs, pool<T>::kernel_stochastic_pool);
        }
        index_kernel = PyTuple_GetItem(output_tuple, 1);
    }
    Py_INCREF(index_kernel);
    // set output
    PyObject* output = PyTuple_GetItem(output_tuple, 0);
    Py_INCREF(output);
    Py_XDECREF(output_tuple);
    return Py_BuildValue("(NNN)", output, index_mask, index_kernel);
}

template<typename T, typename grad_fn>
static PyObject* unpool(PyObject* self, PyObject* args, PyObject* kwargs)
{   
    // apply unpooling functions
    PyObject* output = kernel_unpool<T, grad_fn>(args, kwargs, pool<T>::kernel_unpool);
    return Py_BuildValue("(N)", output);
}

typedef void (rf_fn)(const float*, const float*, size_t, const float*, float*, size_t*, bool);
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