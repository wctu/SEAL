#include <Python.h>
#include <iostream>
#include <cstring>  // memset
using namespace std;

#if PY_MAJOR_VERSION >= 3
long (*PyInt_AsLong)(PyObject *obj) = PyLong_AsLong;
#endif

static PyObject* computeASA(PyObject* self, PyObject* args) {
    PyObject* pListSP;      // input
    PyObject* pListGT;      // input
    PyObject* pListError;   // output
    PyObject* pOutputObject;   // output
    PyObject* pItem;
    int returnErrorMap = 0;
    Py_ssize_t nr_pixel_sp;
    Py_ssize_t nr_pixel_gt;

    // Parse Python args
    if(!PyArg_ParseTuple(args, "OO|i", &pListSP, &pListGT, &returnErrorMap)) {
        PyErr_SetString(PyExc_TypeError, "Arg: SP_label_list, GT_label_list, returnErrorMap=0");
        return NULL;
    }

    // Get input image from Python list to a double array
    nr_pixel_sp = PyList_Size(pListSP);
    nr_pixel_gt = PyList_Size(pListGT);
    if(nr_pixel_sp != nr_pixel_gt) {
        PyErr_SetString(PyExc_TypeError, "Number of pixels mismatched");
        printf("nr_pixel_sp = %d\n", int(nr_pixel_sp));
        printf("nr_pixel_gt = %d\n", int(nr_pixel_gt));
        return NULL;
    }

    int nr_pixel = int(nr_pixel_sp);

    int* SP = new int[nr_pixel];
    int* GT = new int[nr_pixel];

    for(int i = 0; i < nr_pixel; ++i) {
        pItem = PyList_GetItem(pListSP, i);
        SP[i] = PyInt_AsLong(pItem);
        pItem = PyList_GetItem(pListGT, i);
        GT[i] = PyInt_AsLong(pItem);
    }

    // Find number of superpixels and groundtruth segments
    int nr_sp = 0;
    int nr_gt = 0;
    for(int i = 0; i < nr_pixel; ++i) {
        if(SP[i] > nr_sp) nr_sp = SP[i];
        if(GT[i] > nr_gt) nr_gt = GT[i];
    }
    ++nr_sp;
    ++nr_gt;

    double asa = 0.0;
    int* sp_hist = new int[nr_sp*nr_gt];    // a 2D table used to record SP and GT overlaps
    int* sp_size = new int[nr_sp];
    int* max_overlap_size = new int[nr_sp];
    // int* max_overlap_seg = new int[nr_sp];
    memset(sp_hist, 0, sizeof(int)*nr_sp*nr_gt);
    memset(sp_size, 0, sizeof(int)*nr_sp);
    for(int i = 0; i < nr_pixel; ++i) {
        sp_hist[SP[i]*nr_gt + GT[i]] += 1;
        sp_size[SP[i]] += 1;
    }
    for(int i = 0; i < nr_sp; ++i) {
        max_overlap_size[i] = 0;
        // max_overlap_seg[i] = 0;
        for(int j = 0; j < nr_gt; ++j) {
            if(sp_hist[i*nr_gt+j] > max_overlap_size[i]) {
                max_overlap_size[i] = sp_hist[i*nr_gt+j];
                // max_overlap_seg[i] = j;
            }
        }
        asa += double(max_overlap_size[i]);
    }
    asa /= double(nr_pixel);

    if(returnErrorMap) {
        int leakage;
        pListError = PyList_New(nr_pixel);
        for(int i = 0; i < nr_pixel; ++i) {
            leakage = sp_size[SP[i]] - max_overlap_size[SP[i]];
            pItem = PyFloat_FromDouble(double(leakage));
            PyList_SetItem(pListError, i, pItem);
        }
        pOutputObject = Py_BuildValue("dO", asa, pListError);
    }
    else {
        pOutputObject = Py_BuildValue("d", asa);
    }

    delete[] SP;
    delete[] GT;
    delete[] sp_hist;
    delete[] sp_size;
    delete[] max_overlap_size;
    // delete[] max_overlap_seg;

    return pOutputObject;
}

static PyObject* computeBR(PyObject* self, PyObject* args) {
    PyObject* pListSP;      // input
    PyObject* pListGT;      // input
    PyObject* pItem;
    Py_ssize_t nr_pixel_sp;
    Py_ssize_t nr_pixel_gt;

    int h, w, nr_pixel;
    int r = 1;

    // Parse Python args
    if(!PyArg_ParseTuple(args, "OOii|i", &pListSP, &pListGT, &h, &w, &r)) {
        PyErr_SetString(PyExc_TypeError, "Arg: SP_label_list, GT_label_list, height, width, r=1");
        return NULL;
    }

    // Get input image from Python list to a double array
    nr_pixel = h*w;
    nr_pixel_sp = PyList_Size(pListSP);
    nr_pixel_gt = PyList_Size(pListGT);
    if(nr_pixel_sp != nr_pixel_gt) {
        PyErr_SetString(PyExc_TypeError, "Number of pixels mismatched");
        printf("nr_pixel_sp = %d\n", int(nr_pixel_sp));
        printf("nr_pixel_gt = %d\n", int(nr_pixel_gt));
        return NULL;
    }
    if(nr_pixel != nr_pixel_gt) {
        PyErr_SetString(PyExc_TypeError, "Size mismatched");
        printf("nr_pixel_sp = %d\n", int(nr_pixel_sp));
        printf("h*w = %d\n", nr_pixel);
        return NULL;
    }

    int* SP = new int[nr_pixel];
    int* GT = new int[nr_pixel];

    for(int i = 0; i < nr_pixel; ++i) {
        pItem = PyList_GetItem(pListSP, i);
        SP[i] = int(PyInt_AsLong(pItem));
        pItem = PyList_GetItem(pListGT, i);
        GT[i] = int(PyInt_AsLong(pItem));
    }

    unsigned char* transitionSP = new unsigned char[nr_pixel];
    unsigned char* transitionGT = new unsigned char[nr_pixel];
    memset(transitionSP, 0, sizeof(unsigned char)*nr_pixel);
    memset(transitionGT, 0, sizeof(unsigned char)*nr_pixel);
    int index = 0;
    for(int y = 0; y < h; ++y) {
        for(int x = 0; x < w; ++x) {
            if(x < w-1) {
                if(SP[index] != SP[index+1]) transitionSP[index] = 1;
                if(GT[index] != GT[index+1]) transitionGT[index] = 1;
            }
            if(y < h-1) {
                if(SP[index] != SP[index+w]) transitionSP[index] = 1;
                if(GT[index] != GT[index+w]) transitionGT[index] = 1;
            }
            ++index;
        }
    }

    int nr_transition_gt = 0;
    int nr_true_positive = 0;
    bool hit;
    index = 0;
    for(int y = 0; y < h; ++y) {
        for(int x = 0; x < w; ++x) {
            if(transitionGT[index]) {
                ++nr_transition_gt;
                hit = false;
                for(int u = -r; u <= r; ++u) {
                    for(int v = -r; v <= r; ++v) {
                        if(y+u >= 0 && y+u < h && x+v >= 0 && x+v < w) {
                            if(transitionSP[index+u*w+v] > 0) {
                                hit = true;
                                break;
                            }
                        }
                    }
                    if(hit) break;
                }
                if(hit) ++nr_true_positive;
            }
            ++index;
        }
    }

    double br;
    if(nr_transition_gt)
        br = double(nr_true_positive)/double(nr_transition_gt);
    else
        br = 1.0;

    delete[] SP;
    delete[] GT;
    delete[] transitionSP;
    delete[] transitionGT;

    return PyFloat_FromDouble(br);
}

static PyMethodDef EvalSPMethods[] = {
    {
        "computeASA",
        computeASA,
        METH_VARARGS,
        ""
    },
    {
        "computeBR",
        computeBR,
        METH_VARARGS,
        ""
    },
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef EvalSPModule = {
    PyModuleDef_HEAD_INIT,
    "EvalSPModule",
    NULL,
    -1,
    EvalSPMethods
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_EvalSPModule(void) {
     return PyModule_Create(&EvalSPModule);
}
#else
PyMODINIT_FUNC initEvalSPModule(void) {
    (void)Py_InitModule("EvalSPModule", EvalSPMethods);
}
#endif
