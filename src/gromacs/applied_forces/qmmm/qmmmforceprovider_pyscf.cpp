/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2021- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Implements force provider for QMMM
 *
 * \author Dmitry Morozov <dmitry.morozov@jyu.fi>
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#include "gmxpre.h"

#include "qmmmforceprovider.h"

#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/stringutil.h"

// debug, delete later TODO
#include <cstdio>
#include <iostream>
#include <fstream>

namespace gmx
{

QMMMForceProvider::QMMMForceProvider(const QMMMParameters& parameters,
                                     const LocalAtomSet&   localQMAtomSet,
                                     const LocalAtomSet&   localMMAtomSet,
                                     PbcType               pbcType,
                                     const MDLogger&       logger) :
    parameters_(parameters),
    qmAtoms_(localQMAtomSet),
    mmAtoms_(localMMAtomSet),
    pbcType_(pbcType),
    logger_(logger),
    box_{ { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } }
{
}

QMMMForceProvider::~QMMMForceProvider()
{
    //Py_Finalize();
    if (force_env_ != -1)
    {
        //cp2k_destroy_force_env(force_env_);
        if (GMX_LIB_MPI)
        {
            // Python finalize?
            //cp2k_finalize_without_mpi();
        }
        else
        {
            // Python finalize?
            //cp2k_finalize();
        }
    }
}

bool QMMMForceProvider::isQMAtom(Index globalAtomIndex)
{
    return std::find(qmAtoms_.globalIndex().begin(), qmAtoms_.globalIndex().end(), globalAtomIndex)
           != qmAtoms_.globalIndex().end();
}

void QMMMForceProvider::appendLog(const std::string& msg)
{
    GMX_LOG(logger_.info).asParagraph().appendText(msg);
}

void QMMMForceProvider::initPython(const t_commrec& cr)
{
    // Currently python initialization only works in the main function.
    // Need to find a proper place.
    // If I place them here, import_array 100% fails at longdouble_multiply().
    //Py_Initialize();
    /*
    import_array1(void(0));
    if (PyErr_Occurred()) {
        fprintf(stderr, "Failed to import numpy Python module(s).\n");
        return;
    }
    assert(PyArray_API);
    */

    // Set flag of successful initialization
    isPythonInitialized_ = true;
} // namespace gmx

void QMMMForceProvider::calculateForces(const ForceProviderInput& fInput, ForceProviderOutput* fOutput)
{
    if (!isPythonInitialized_)
    {
        try
        {
            initPython(fInput.cr_);
        }
        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR;
    }
    fprintf(stderr, "Importing pyscfdriverii...\n");
    PyObject* pModule = PyImport_ImportModule("pyscfdriverii");
    if (!pModule) {
        fprintf(stderr, "pyscfdriverii load failed!\n");
        exit(1);
    }
    fprintf(stderr, "pyscfdriverii load successful!\n");
    // PyObject* pFunc = PyObject_GetAttrString(pModule, "QMcalculation");
    // PyObject* pFuncPrint = PyObject_GetAttrString(pModule, "printProp");
    fprintf(stderr, "Loading qmmmCalc...\n");
    PyObject* pFuncCalc = PyObject_GetAttrString(pModule, "qmmmCalc");
    if (!pFuncCalc) {
        fprintf(stderr, "qmmmCalc load failed!\n");
        exit(1);
    }
    fprintf(stderr, "qmmmCalc load successful!\n");

    // Total number of atoms in the system
    size_t numAtoms = qmAtoms_.numAtomsGlobal() + mmAtoms_.numAtomsGlobal();
    size_t numAtomsQM = qmAtoms_.numAtomsGlobal();
    size_t numAtomsMM = mmAtoms_.numAtomsGlobal();
    // Save box
    copy_mat(fInput.box_, box_);

    // Initialize PBC
    t_pbc pbc;
    set_pbc(&pbc, pbcType_, box_);
    /*
     * 1) We need to gather fInput.x_ in case of MPI / DD setup
     */

    // x - coordinates (gathered across nodes in case of DD)
    std::vector<RVec> x(numAtoms, RVec({ 0.0, 0.0, 0.0 }));

    // Fill cordinates of local QM atoms and add translation
    PyObject* pyQMKinds = PyList_New(numAtomsQM);
    PyObject* pyQMCoords = PyList_New(numAtomsQM);

    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {
        x[qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]] =
                fInput.x_[qmAtoms_.localIndex()[i]] + parameters_.qmTrans_;

        PyObject* pySymbol = PyUnicode_FromString(periodic_system[parameters_.atomNumbers_[qmAtoms_.localIndex()[i]]].c_str());
        PyList_SetItem(pyQMKinds, i, pySymbol);

        PyObject* pyCoords_row = PyList_New(3);
        double coordTempX =  10 * (fInput.x_[qmAtoms_.localIndex()[i]][XX]  + parameters_.qmTrans_[XX]);
        double coordTempY =  10 * (fInput.x_[qmAtoms_.localIndex()[i]][YY]  + parameters_.qmTrans_[YY]);
        double coordTempZ =  10 * (fInput.x_[qmAtoms_.localIndex()[i]][ZZ]  + parameters_.qmTrans_[ZZ]);

        PyObject* pyCoordsX = PyFloat_FromDouble(coordTempX);
        PyObject* pyCoordsY = PyFloat_FromDouble(coordTempY);
        PyObject* pyCoordsZ = PyFloat_FromDouble(coordTempZ);

        PyList_SetItem(pyCoords_row, XX, pyCoordsX);
        PyList_SetItem(pyCoords_row, YY, pyCoordsY);
        PyList_SetItem(pyCoords_row, ZZ, pyCoordsZ);

        PyList_SetItem(pyQMCoords, i, pyCoords_row);
    }

    // PyObject_CallFunctionObjArgs(pFuncPrint, pyQMKinds, NULL);
    // PyObject_CallFunctionObjArgs(pFuncPrint, pyQMCoords, NULL);

    // Fill cordinates of local MM atoms and add translation
    PyObject* pyMMKinds = PyList_New(numAtomsMM);
    PyObject* pyMMCharges = PyList_New(numAtomsMM);
    PyObject* pyMMCoords = PyList_New(numAtomsMM);
    for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
    {
        x[mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]] =
                fInput.x_[mmAtoms_.localIndex()[i]] + parameters_.qmTrans_;

        PyObject* pySymbol = PyUnicode_FromString(periodic_system[parameters_.atomNumbers_[mmAtoms_.localIndex()[i]]].c_str());
        PyList_SetItem(pyMMKinds, i, pySymbol);

        PyObject* pyCharge = PyFloat_FromDouble(fInput.chargeA_[mmAtoms_.localIndex()[i]]);
        PyList_SetItem(pyMMCharges, i, pyCharge);

        PyObject* pyCoords_row = PyList_New(3);
        double coordTempX =  10 * (fInput.x_[mmAtoms_.localIndex()[i]][XX]  + parameters_.qmTrans_[XX]);
        double coordTempY =  10 * (fInput.x_[mmAtoms_.localIndex()[i]][YY]  + parameters_.qmTrans_[YY]);
        double coordTempZ =  10 * (fInput.x_[mmAtoms_.localIndex()[i]][ZZ]  + parameters_.qmTrans_[ZZ]);

        PyObject* pyCoordsX = PyFloat_FromDouble(coordTempX);
        PyObject* pyCoordsY = PyFloat_FromDouble(coordTempY);
        PyObject* pyCoordsZ = PyFloat_FromDouble(coordTempZ);

        PyList_SetItem(pyCoords_row, XX, pyCoordsX);
        PyList_SetItem(pyCoords_row, YY, pyCoordsY);
        PyList_SetItem(pyCoords_row, ZZ, pyCoordsZ);

        PyList_SetItem(pyMMCoords, i, pyCoords_row);

    }

    // PyObject_CallFunctionObjArgs(pFuncPrint, pyMMKinds, NULL);
    // PyObject_CallFunctionObjArgs(pFuncPrint, pyMMCharges, NULL);
    // PyObject_CallFunctionObjArgs(pFuncPrint, pyMMCoords, NULL);

    // If we are in MPI / DD conditions then gather coordinates over nodes
    if (havePPDomainDecomposition(&fInput.cr_))
    {
        gmx_sum(3 * numAtoms, x.data()->as_vec(), &fInput.cr_);
    }

    // Put all atoms into the central box (they might be shifted out of it because of the translation)
    put_atoms_in_box(pbcType_, fInput.box_, ArrayRef<RVec>(x));


    const std::string qm_basis = "ccpvdz";

    // TODO: fix for MPI version

    PyObject* pQMBasis = PyUnicode_FromString(qm_basis.c_str());
    PyObject* pQMMult = PyLong_FromLong(parameters_.qmMultiplicity_);
    PyObject* pQMCharge = PyLong_FromLong(parameters_.qmCharge_);

    PyObject* pyscfCalcReturn = PyObject_CallFunctionObjArgs(pFuncCalc,
            pQMBasis, pQMMult, pQMCharge,
            pyQMKinds, pyQMCoords, pyMMKinds,
            pyMMCharges, pyMMCoords, NULL);
    if (!pyscfCalcReturn){
        fprintf(stderr, "pyscfCalcReturn IS nullptr!!!\n");
    }

    PyObject* pQMMMEnergy = PyTuple_GetItem(pyscfCalcReturn, 0);
    PyObject* pQMForce = PyTuple_GetItem(pyscfCalcReturn, 1);
    PyObject* pMMForce = PyTuple_GetItem(pyscfCalcReturn, 2);

    // fprintf(stdout, "Python print test QMForce\n");
    // PyObject_CallFunctionObjArgs(pFuncPrint, pQMForce, NULL);

    double qmmmEnergy(0);
    if (pQMMMEnergy) {
        qmmmEnergy = PyFloat_AsDouble(pQMMMEnergy);
        fprintf(stderr, "GROMACS received energy %f \n", qmmmEnergy);
    } else {
        fprintf(stderr, "pointer to pyscf returned energy is nullptr\n");
    }

    if (!pQMForce){
        fprintf(stderr, "parsing pyscfCalcReturn error, pyobject pQMForce is nullptr\n");
    }

    if (!pMMForce){
        fprintf(stderr, "parsing pyscfCalcReturn error, pyobject pMMForce is nullptr\n");
    }

    PyArrayObject* npyQMForce = reinterpret_cast<PyArrayObject*>(pQMForce);

    npy_intp npyQMForce_rows = PyArray_DIM(npyQMForce, 0);
    npy_intp npyQMForce_columns = PyArray_DIM(npyQMForce, 1);
    int qm_num = static_cast<int>(npyQMForce_rows);
    int qm_coordDIM = static_cast<int>(npyQMForce_columns);
    fprintf(stderr, "qm_num (%d), qm_coordDIM (%d), from taking the size of pyobject \n", qm_num, qm_coordDIM);
    double qmForce[qm_num*qm_coordDIM] = {};
    double* npyQMForce_cast = static_cast<double*>(PyArray_DATA(npyQMForce));

    for (npy_intp i = 0; i < npyQMForce_rows; ++i) {
        for (npy_intp j = 0; j < npyQMForce_columns; ++j) {
            qmForce[i*3+j] = npyQMForce_cast[i * npyQMForce_columns + j];
        }
    }
    PyArrayObject* npyMMForce = reinterpret_cast<PyArrayObject*>(pMMForce);

    npy_intp npyMMForce_rows = PyArray_DIM(npyMMForce, 0);
    npy_intp npyMMForce_columns = PyArray_DIM(npyMMForce, 1);

    int mm_num = static_cast<int>(npyMMForce_rows);
    int mm_coordDIM = static_cast<int>(npyMMForce_columns);
    double mmForce[mm_num*mm_coordDIM] = {};
    double* npyMMForce_cast = static_cast<double*>(PyArray_DATA(npyMMForce));

    for (npy_intp i = 0; i < npyMMForce_rows; ++i) {
        for (npy_intp j = 0; j < npyMMForce_columns; ++j) {
            mmForce[i*3+j] = npyMMForce_cast[i * npyMMForce_columns + j];
        }
    }
    // fprintf(stdout, "qm_num: %d, mm_num: %d\nqm_coordDIM: %d, mm_coordDIM: %d\nnumAtoms: %d\n", qm_num, mm_num, qm_coordDIM, mm_coordDIM, static_cast<int>(numAtoms));

    /*
     * 2) Cast data to double format of libpython
     *    update coordinates and box in PySCF and perform QM calculation
     */
    // x_d - coordinates casted to linear dobule vector for PySCF with parameters_.qmTrans_ added
    std::vector<double> x_d(3 * numAtoms, 0.0);
    for (size_t i = 0; i < numAtoms; i++)
    {
        x_d[3 * i]     = static_cast<double>((x[i][XX]) / c_bohr2Nm);
        x_d[3 * i + 1] = static_cast<double>((x[i][YY]) / c_bohr2Nm);
        x_d[3 * i + 2] = static_cast<double>((x[i][ZZ]) / c_bohr2Nm);
    }

    // box_d - box_ casted to linear dobule vector for PySCF
    std::vector<double> box_d(9);
    for (size_t i = 0; i < DIM; i++)
    {
        box_d[3 * i]     = static_cast<double>(box_[0][i] / c_bohr2Nm);
        box_d[3 * i + 1] = static_cast<double>(box_[1][i] / c_bohr2Nm);
        box_d[3 * i + 2] = static_cast<double>(box_[2][i] / c_bohr2Nm);
    }

    // NOTE: need to handle MPI
    // Update coordinates and box in PySCF
    //cp2k_set_positions(force_env_, x_d.data(), 3 * numAtoms);
    //cp2k_set_cell(force_env_, box_d.data());
    // Check if we have external MPI library
    if (GMX_LIB_MPI)
    {
        // We have an external MPI library
#if GMX_LIB_MPI
#endif
    }
    else
    {
        // If we have thread-MPI or no-MPI then we should initialize CP2P differently
    }

    // Run PySCF calculation
    //cp2k_calc_energy_force(force_env_);

    /*
     * 3) Get output data
     * We need to fill only local part into fOutput
     */

    // Only main process should add QM + QMMM energy
    if (MAIN(&fInput.cr_))
    {
        double qmEner = 0.0;
        qmEner = qmmmEnergy;
        //cp2k_get_potential_energy(force_env_, &qmEner);
        fOutput->enerd_.term[F_EQM] += qmEner * c_hartree2Kj * c_avogadro;
    }

    // Get Forces they are in Hartree/Bohr and will be converted to kJ/mol/nm
    std::vector<double> pyscfForce(3 * numAtoms, 0.0);

    // Fill forces on QM atoms first
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]]
                     = qmForce[qmAtoms_.collectiveIndex()[i] * 3 + 0];
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 1]
                     = qmForce[qmAtoms_.collectiveIndex()[i] * 3 + 1];
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 2]
                     = qmForce[qmAtoms_.collectiveIndex()[i] * 3 + 2];

        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][XX] +=
                static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][YY] +=
                static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 1])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][ZZ] +=
                static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 2])
                * c_hartreeBohr2Md;
    }

    // Fill forces on MM atoms then
    for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
    {
        pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]]
                     = mmForce[mmAtoms_.collectiveIndex()[i] * 3 + 0];
        pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 1]
                     = mmForce[mmAtoms_.collectiveIndex()[i] * 3 + 1];
        pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 2]
                     = mmForce[mmAtoms_.collectiveIndex()[i] * 3 + 2];

        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][XX] +=
                static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][YY] +=
                static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 1])
                * c_hartreeBohr2Md;

        fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][ZZ] +=
                static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 2])
                * c_hartreeBohr2Md;
    }

    forceRecorder(fOutput, pyscfForce, fInput);

    Py_XDECREF(pFuncCalc);
    // Py_XDECREF(pFunc);
    // Py_XDECREF(pFuncPrint);
    Py_XDECREF(pModule);
    // Py_XDECREF(pPDBname);
    // Py_XDECREF(pQMBasis);
    // Py_XDECREF(pQMMult);
    // Py_XDECREF(pyscfReturn);
    // Py_XDECREF(pQMCharge);
    // Py_XDECREF(pQMMMEnergy);
    // Py_XDECREF(pQMForce);
    // Py_XDECREF(pMMForce);
    // fprintf(stdout, "test1!!!\n");
    // Py_XDECREF(npyQMForce);
    // Py_XDECREF(npyQMForce_rows);
    // Py_XDECREF(npyQMForce_columns);
    // Py_XDECREF(npyMMForce);
    // Py_XDECREF(npyMMForce_rows);
    // Py_XDECREF(npyMMForce_columns);
    // fprintf(stdout, "test_before_py_finalize\n");
    // Py_FinalizeEx();
    // fprintf(stdout, "test_after_py_finalize\n");

};


void QMMMForceProvider::forceRecorder(ForceProviderOutput* fOutput, std::vector<double> pyscfForce, const ForceProviderInput& fInput)
{

    std::string QMMM_record = "";
    std::ofstream recordFile;
    recordFile.open("record_pyscf.txt", std::ios::app);
    // size_t numAtoms = qmAtoms_.numAtomsLocal() + mmAtoms_.numAtomsLocal();
    // recordFile << "number of atoms" << numAtoms << std::endl;
    // recordFile << "step number = " << "need to find..." << std::endl;

    double Fx = 0.0, Fy = 0.0 , Fz = 0.0;
    double Ftotx = 0.0, Ftoty = 0.0 , Ftotz = 0.0;
    double coordx = 0.0, coordy = 0.0, coordz = 0.0;
    double charge = 0.0;
    QMMM_record += formatString("step = ");
    QMMM_record += formatString("%" PRId64, fInput.step_);
    QMMM_record += formatString(", time = %f\n", fInput.t_);
    QMMM_record += formatString("   %7s %7s %7s %12s %9s %9s %9s %9s %9s %10s %4s %6s %6s %6s\n", "x", "y", "z", "Fqmmmmx", "Fqmmmmy", "Fqmmmz", "Ftotx", "Ftoty", "Ftotz", "charge", "i","local", "global", "collec");
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {

        QMMM_record += formatString("%2s QM  ", periodic_system[parameters_.atomNumbers_[qmAtoms_.globalIndex()[i]]].c_str());
        Ftotx = fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][XX]/c_hartreeBohr2Md;
        Ftoty = fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][YY]/c_hartreeBohr2Md;
        Ftotz = fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][ZZ]/c_hartreeBohr2Md;
        Fx = static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]]);
        Fy = static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 1]);
        Fz = static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 2]);

        // Fx = Fx/c_bohr2ANG;
        // Fy = Fy/c_bohr2ANG;
        // Fz = Fz/c_bohr2ANG;

        coordx = (fInput.x_[qmAtoms_.localIndex()[i]][XX] + parameters_.qmTrans_[XX]) * 10;
        coordy = (fInput.x_[qmAtoms_.localIndex()[i]][YY] + parameters_.qmTrans_[YY]) * 10;
        coordz = (fInput.x_[qmAtoms_.localIndex()[i]][ZZ] + parameters_.qmTrans_[ZZ]) * 10;

        charge = fInput.chargeA_[qmAtoms_.localIndex()[i]];

        QMMM_record += formatString("%7.4lf %7.4lf %7.4lf  ", coordx, coordy, coordz);
        QMMM_record += formatString("%9.5lf %9.5lf %9.5lf ", Fx, Fy, Fz);
        QMMM_record += formatString("%9.5lf %9.5lf %9.5lf", Ftotx, Ftoty, Ftotz);
        QMMM_record += formatString("%7.3f %4d %4d %4d %4d\n", 0.0, static_cast<int>(i), qmAtoms_.localIndex()[i], qmAtoms_.globalIndex()[i], qmAtoms_.collectiveIndex()[i]);
    }


    for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
    {

        QMMM_record += formatString("%2s MM  ", periodic_system[parameters_.atomNumbers_[mmAtoms_.globalIndex()[i]]].c_str());
        Ftotx = fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][XX]/c_hartreeBohr2Md;
        Ftoty = fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][YY]/c_hartreeBohr2Md;
        Ftotz = fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][ZZ]/c_hartreeBohr2Md;
        Fx = static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]]);
        Fy = static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 1]);
        Fz = static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 2]);

        // Fx = Fx/c_bohr2ANG;
        // Fy = Fy/c_bohr2ANG;
        // Fz = Fz/c_bohr2ANG;

        coordx = (fInput.x_[mmAtoms_.localIndex()[i]][XX] + parameters_.qmTrans_[XX]) * 10;
        coordy = (fInput.x_[mmAtoms_.localIndex()[i]][YY] + parameters_.qmTrans_[XX]) * 10;
        coordz = (fInput.x_[mmAtoms_.localIndex()[i]][ZZ] + parameters_.qmTrans_[XX]) * 10;

        charge = fInput.chargeA_[mmAtoms_.localIndex()[i]];
        QMMM_record += formatString("%7.4lf %7.4lf %7.4lf  ", coordx, coordy, coordz);
        QMMM_record += formatString("%9.5lf %9.5lf %9.5lf ", Fx, Fy, Fz);
        QMMM_record += formatString("%9.5lf %9.5lf %9.5lf", Ftotx, Ftoty, Ftotz);
        QMMM_record += formatString("%7.3f %4d %4d %4d %4d\n", charge, static_cast<int>(i), mmAtoms_.localIndex()[i], mmAtoms_.globalIndex()[i], mmAtoms_.collectiveIndex()[i]);
    }

    QMMM_record += formatString("\n");

    recordFile << QMMM_record;

    recordFile.close();

    return;
}


} // namespace gmx
