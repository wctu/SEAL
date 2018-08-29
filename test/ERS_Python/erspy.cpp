#include <Python.h>

#include <iostream>
#include <string>
#include <cmath>
#include <stdlib.h>

#include "MERCLazyGreedy.h"
#include "MERCInputImage.h"
#include "MERCOutputImage.h"
#include "Image.h"
#include "ImageIO.h"

// Original ERS
static PyObject* ERS(PyObject* self, PyObject* args) {
    PyObject* pListImg;
    PyObject* pListLabel;
    PyObject* pItem;
    Py_ssize_t nr_pixel;

    int height;
    int width;
    int nr_segment = 300;    // number of superpixels
    double lambda = 0.5;
    double sigma = 5.0;
    int conn8 = 0;
    int kernel = 0;     // kernel_type 0 is Gaussian
    int compute_similarity = 1;

    // Parse Python args
    if(!PyArg_ParseTuple(args, "Oiii|idd", &pListImg, &height, &width, &nr_segment, &conn8, &lambda, &sigma)) {
        PyErr_SetString(PyExc_TypeError, "Arg: img_list, h, w, nC, conn8=0, lambda=0.5, sigma=5.0");
        return NULL;
    }

    // Get input image from Python list to a double array
    nr_pixel = PyList_Size(pListImg);
    double* img = new double[nr_pixel];
    for(int i = 0; i < nr_pixel; ++i) {
        pItem = PyList_GetItem(pListImg, i);
        img[i] = PyFloat_AsDouble(pItem);
    }

    // Load the image to correct buffer
    Image<RGBMap> inputImage;
    MERCInputImage<RGBMap> input;
    inputImage.Resize(width, height, false);
    int idx = 0;
    for(int y = 0; y < height; ++y) {
        for(int x = 0; x < width; ++x) {
            RGBMap color(
            (int)img[idx++],
            (int)img[idx++],
            (int)img[idx++]);
            inputImage.Access(x, y) = color;
        }
    }
    input.ReadImage(&inputImage, conn8);
    
    // ERS segmentation
    MERCLazyGreedy merc;
    merc.ClusteringTreeIF(input.nNodes_, input, kernel, sigma*3, lambda*1.0*nr_segment, nr_segment, compute_similarity);
    vector<int> label = MERCOutputImage::DisjointSetToLabel(merc.disjointSet_);

    // Output
    pListLabel = PyList_New(height*width);
    for(int i = 0; i < height*width; ++i) {
        pItem = PyInt_FromLong(label[i]);
        PyList_SetItem(pListLabel, i, pItem);
    }

    delete[] img;

    return pListLabel;
}

// Modified ERS
static PyObject* ERSWgtOnly(PyObject* self, PyObject* args) {

    int height;
    int width;
    int nC;    // number of superpixels
    double lambda = 0.5;
    
    int conn8 = 0;      // must be 0 for this version
    int compute_similarity = 0;

    // These are not used but still need to be declared here
    double sigma = 5.0;
    int kernel = 0; 

    // Parse Python args
    PyObject* pListAffinity;
    PyObject* pListLabel;
    PyObject* pItem;
    Py_ssize_t nr_wgt;
    if(!PyArg_ParseTuple(args, "Oiii|id", &pListAffinity, &height, &width, &nC, &conn8, &lambda)) {
        PyErr_SetString(PyExc_TypeError, "Arg: wgt_list, h, w, nC, conn8=0, lambda=0.5");
        return NULL;
    }

    int nr_pixel = height*width;

    // Load weight
    nr_wgt = PyList_Size(pListAffinity); // will be w*h*2
    if(nr_wgt != height*width*2) {
        printf("height = %d\nwidth = %d\nnr_wgt = %d\n", height, width, nr_wgt);
        PyErr_SetString(PyExc_TypeError, "Something wrong with weight list.");
        return NULL;
    }

    double* affinity_x = new double[nr_pixel];
    double* affinity_y = new double[nr_pixel];
    for(int i = 0; i < nr_pixel; ++i) {
        pItem = PyList_GetItem(pListAffinity, i);
        affinity_x[i] = PyFloat_AsDouble(pItem);
        pItem = PyList_GetItem(pListAffinity, i + nr_pixel);
        affinity_y[i] = PyFloat_AsDouble(pItem);
    }

    // Load the image to correct buffer
    MERCInputImage<RGBMap> input;
    input.ReadImageWgt(affinity_x, affinity_y, height, width, conn8);
    
    // ERS segmentation
    MERCLazyGreedy merc;
    merc.ClusteringTreeIF(input.nNodes_, input, kernel, sigma*3, lambda*nC, nC, compute_similarity);
    vector<int> label = MERCOutputImage::DisjointSetToLabel(merc.disjointSet_);

    // Output
    pListLabel = PyList_New(height*width);
    for(int i = 0; i < height*width; ++i) {
        pItem = PyInt_FromLong(label[i]);
        PyList_SetItem(pListLabel, i, pItem);
    }

    delete [] affinity_x;
    delete [] affinity_y;

    return pListLabel;
}

static PyMethodDef ERSMethods[] = {
    {
        "ERS",
        ERS,
        METH_VARARGS,
        ""
    },
    {
        "ERSWgtOnly",
        ERSWgtOnly,
        METH_VARARGS,
        ""
    },
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initERSModule(void) {
    (void)Py_InitModule("ERSModule", ERSMethods);
}
