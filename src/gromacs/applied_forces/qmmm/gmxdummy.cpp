#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"

void call_python(void);

int main()
{
    int step_total = 3;

    Py_Initialize();
    _import_array();

    for (int step = 0; step < step_total; step++){
        fprintf(stdout, "step %d\n", (step+1));
        call_python();
    }
    Py_Finalize();
    return 0;
}

void call_python(void){
        PyObject *pPDBname = PyUnicode_FromString("_cp2k_HFHOH.pdb");
        PyObject *pQMBasis = PyUnicode_FromString("ccpvdz");
        PyObject *pQMMult = PyLong_FromLong(1);
        PyObject *pQMCharge = PyLong_FromLong(0);

        PyObject *pModule = PyImport_ImportModule("pyscfdriverii");
        if (pModule){
            PyObject *pFunc = PyObject_GetAttrString(pModule, "QMcalculation");
            if (pFunc){
                PyObject *pReturn = PyObject_CallFunctionObjArgs(pFunc,
                            pPDBname, pQMBasis, pQMMult, pQMCharge, NULL);


                PyObject *pQMEnergy = PyTuple_GetItem(pReturn, 0);
                PyObject *pQMForce = PyTuple_GetItem(pReturn, 1);
                PyObject *pMMForce = PyTuple_GetItem(pReturn, 2);

                double qmmmEnergy(0);
                qmmmEnergy = PyFloat_AsDouble(pQMEnergy);
                std::cout << "C++ received energy = " << std::setprecision(15) << qmmmEnergy << std::endl;

                // if (PyArray_Check(pQMForce)) {
                PyArrayObject *npyQMForce = reinterpret_cast<PyArrayObject*>(pQMForce);

                npy_intp npyQMForce_rows = PyArray_DIM(npyQMForce, 0);
                npy_intp npyQMForce_columns = PyArray_DIM(npyQMForce, 1);
                int qm_num = static_cast<int>(npyQMForce_rows);
                int qm_coordDIM = static_cast<int>(npyQMForce_columns);
                double qm_force[qm_num*qm_coordDIM] = {};
                double *npyQMForce_cast = static_cast<double*>(PyArray_DATA(npyQMForce));

                for (npy_intp i = 0; i < npyQMForce_rows; ++i) {
                    for (npy_intp j = 0; j < npyQMForce_columns; ++j) {
                        qm_force[i*3 + j] = npyQMForce_cast[i * npyQMForce_columns + j];
                    }
                }
                std::cout << "   qm_kind   " <<  "      Fx      " << "      Fy      " << "      Fz      " << std::endl;
                for (int i = 0; i < qm_num; i ++)
                {
                    std::cout << "  " << std::setprecision(10) << qm_force[i*3] << "  ";
                    std::cout << "  " << std::setprecision(10) << qm_force[i*3+1] << "  ";
                    std::cout << "  " << std::setprecision(10) << qm_force[i*3+2] << "  " << std::endl;
                }

                // }

                PyArrayObject *npyMMForce = reinterpret_cast<PyArrayObject*>(pMMForce);

                npy_intp npyMMForce_rows = PyArray_DIM(npyMMForce, 0);
                npy_intp npyMMForce_columns = PyArray_DIM(npyMMForce, 1);
                int mm_num = static_cast<int>(npyMMForce_rows);
                int mm_coordDIM = static_cast<int>(npyMMForce_columns);
                double mm_force[mm_num*mm_coordDIM];

                double *npyMMForce_cast = static_cast<double*>(PyArray_DATA(npyMMForce));

                for (npy_intp i = 0; i < npyMMForce_rows; ++i) {
                    for (npy_intp j = 0; j < npyMMForce_columns; ++j) {
                        mm_force[i*3 + j] = npyMMForce_cast[i * npyMMForce_columns + j];
                    }
                }
                std::cout << "   mm_charge   " <<  "      Fx      " << "      Fy      " << "      Fz      " << std::endl;
                for (int i = 0; i < mm_num; i ++)
                {
                    std::cout << "  " << std::setprecision(10) << mm_force[i*3 + 0] << "  ";
                    std::cout << "  " << std::setprecision(10) << mm_force[i*3 + 1] << "  ";
                    std::cout << "  " << std::setprecision(10) << mm_force[i*3 + 2] << "  " << std::endl;
                }
                Py_XDECREF(pPDBname);
                Py_XDECREF(pQMBasis);
                Py_XDECREF(pQMMult);
                Py_XDECREF(pQMCharge);

                PyArray_XDECREF(npyMMForce);
                PyArray_XDECREF(npyQMForce);

                Py_XDECREF(pReturn);
                // Py_SET_REFCNT(pQMEnergy,0);
                std::cout << "pQMEnergy ref count " << Py_REFCNT(pQMEnergy) << std::endl;
                // Py_XDECREF(pQMEnergy);
                // Py_XDECREF(pQMForce);
                // Py_XDECREF(pMMForce);
            }
            std::cout << "pFunc ref count " << Py_REFCNT(pFunc) << std::endl;
            Py_XDECREF(pFunc);
            std::cout << "pFunc ref count " << Py_REFCNT(pFunc) << std::endl;
        }
        Py_XDECREF(pModule);
}
