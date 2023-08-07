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

#include "gmxpre.h"

#include "qmmmforceprovider.h"

#include <Python.h>
#include "numpy/arrayobject.h"

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

namespace
{

/*! \brief Helper function that dumps string into the file.
 *
 * \param[in] filename Name of the file to write.
 * \param[in] str String to write into the file.
 * \throws    std::bad_alloc if out of memory.
 * \throws    FileIOError on any I/O error.
 */
void writeStringToFile(const std::string& filename, const std::string& str)
{

    TextOutputFile fOut(filename);
    fOut.write(str.c_str());
    fOut.close();
}

} // namespace

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

//void QMMMForceProvider::initPyscfForceEnvironment(const t_commrec& cr)
//{
//
//} // namespace gmx

void QMMMForceProvider::calculateForces(const ForceProviderInput& fInput, ForceProviderOutput* fOutput)
{
    // Total number of atoms in the system
    size_t numAtoms = qmAtoms_.numAtomsGlobal() + mmAtoms_.numAtomsGlobal();

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
    Py_Initialize();
    _import_array();
    const std::string qm_basis = "ccpvdz";
    int qm_charge = 0, qm_mult = 1;
    const std::string module_name = "pyscfdriverii";
    const std::string func_name = "QMcalculation";
    std::string pdb_name = parameters_.qmFileNameBase_ + ".pdb";

    // TODO: fix for MPI version
    // Write CP2K input if we are Main
    // Write *.pdb with point charges for CP2K
    writeStringToFile(pdb_name, parameters_.qmPdb_);

    PyObject* pQMBasis = PyUnicode_FromString(qm_basis.c_str());
    PyObject* pQMMult = PyLong_FromLong(qm_mult);
    PyObject* pQMCharge = PyLong_FromLong(qm_charge);
    PyObject* pPDBname = PyUnicode_FromString(pdb_name.c_str());

    // import mymodule
    // PyObject* pName = NULL;
    // pName = PyUnicode_FromString(module_name.c_str());
    // if (pName != NULL)
    // {
    //     fprintf(stderr, "pName is not NULL\n");
    // } else {
    //     fprintf(stderr, "pName IS NULL\n");
    // }
    PyObject* pModule = NULL;
    // pModule = PyImport_Import(pName);
    pModule = PyImport_ImportModule(module_name.c_str());

    // import and call function
    if (pModule != NULL)
    {
        fprintf(stderr, "pModule is not NULL\n");
    } else {
        fprintf(stderr, "pModule IS NULL\n");
    }
    PyObject* pFunc = PyObject_GetAttrString(pModule, func_name.c_str());

    PyObject* pysfcReturn = PyObject_CallFunctionObjArgs(pFunc,
            pPDBname, pQMBasis, pQMMult, pQMCharge, NULL);
    //parse the pyscfReturn into different parts

    PyObject* pQMMMEnergy = PyTuple_GetItem(pysfcReturn, 0);
    PyObject* pQMForce = PyTuple_GetItem(pysfcReturn, 1);
    PyObject* pMMForce = PyTuple_GetItem(pysfcReturn, 2);

    if (pQMMMEnergy != NULL)
    {
        fprintf(stderr, "pQMMMEnergy is not NULL\n");
    } else {
        fprintf(stderr, "pQMMMEnergy IS NULL\n");
    }

    if (pQMForce != NULL)
    {
        fprintf(stderr, "pQMForce is not NULL\n");
    } else {
        fprintf(stderr, "pQMForce IS NULL\n");
    }

    if (pMMForce != NULL)
    {
        fprintf(stderr, "pMMForce is not NULL\n");
    } else {
        fprintf(stderr, "pMMForce IS NULL\n");
    }

    double qmmmEnergy(0);
    qmmmEnergy = PyFloat_AsDouble(pQMMMEnergy);
    fprintf(stderr, "GROMACS received energy %f \n", qmmmEnergy);

    PyArrayObject* npyQMForce = reinterpret_cast<PyArrayObject*>(pQMForce);

    npy_intp npyQMForce_rows = PyArray_DIM(npyQMForce, 0);
    npy_intp npyQMForce_columns = PyArray_DIM(npyQMForce, 1);
    int qm_num = static_cast<int>(npyQMForce_rows);
    int qm_coordDIM = static_cast<int>(npyQMForce_columns);
    double qm_force[qm_num*qm_coordDIM] = {};
    double* npyQMForce_cast = static_cast<double*>(PyArray_DATA(npyQMForce));

    for (npy_intp i = 0; i < npyQMForce_rows; ++i) {
        for (npy_intp j = 0; j < npyQMForce_columns; ++j) {
            qm_force[i*3+j] = npyQMForce_cast[i * npyQMForce_columns + j];
        }
    }
    // if (PyArray_Check(pMMForce)) {
    // }
    PyArrayObject* npyMMForce = reinterpret_cast<PyArrayObject*>(pMMForce);

    npy_intp npyMMForce_rows = PyArray_DIM(npyMMForce, 0);
    npy_intp npyMMForce_columns = PyArray_DIM(npyMMForce, 1);

    int mm_num = static_cast<int>(npyMMForce_rows);
    int mm_coordDIM = static_cast<int>(npyMMForce_columns);
    double mm_force[mm_num*mm_coordDIM] = {};
    double* npyMMForce_cast = static_cast<double*>(PyArray_DATA(npyMMForce));

    for (npy_intp i = 0; i < npyMMForce_rows; ++i) {
        for (npy_intp j = 0; j < npyMMForce_columns; ++j) {
            mm_force[i*3+j] = npyMMForce_cast[i * npyMMForce_columns + j];
        }
    }
    fprintf(stderr, "qm_num: %d, mm_num: %d\nqm_coordDIM: %d, mm_coordDIM: %d, \nnumAtoms: %d\n", qm_num, mm_num, qm_coordDIM, mm_coordDIM, static_cast<int>(numAtoms));

    // Fill cordinates of local QM atoms and add translation
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {
        x[qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]] =
                fInput.x_[qmAtoms_.localIndex()[i]] + parameters_.qmTrans_;
    }

    // Fill cordinates of local MM atoms and add translation
    for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
    {
        x[mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]] =
                fInput.x_[mmAtoms_.localIndex()[i]] + parameters_.qmTrans_;
    }

    // If we are in MPI / DD conditions then gather coordinates over nodes
    if (havePPDomainDecomposition(&fInput.cr_))
    {
        gmx_sum(3 * numAtoms, x.data()->as_vec(), &fInput.cr_);
    }

    // Put all atoms into the central box (they might be shifted out of it because of the translation)
    put_atoms_in_box(pbcType_, fInput.box_, ArrayRef<RVec>(x));


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
        //cp2k_get_potential_energy(force_env_, &qmEner);
        fOutput->enerd_.term[F_EQM] += qmEner * c_hartree2Kj * c_avogadro;
    }

    // Get Forces they are in Hartree/Bohr and will be converted to kJ/mol/nm
    std::vector<double> pyscfForce(3 * numAtoms, 0.0);

    // Fill forces on QM atoms first
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]]
                     = qm_force[3 * qmAtoms_.collectiveIndex()[i]];
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 1]
                     = qm_force[3 * qmAtoms_.collectiveIndex()[i] + 1];
        pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 2]
                     = qm_force[3 * qmAtoms_.collectiveIndex()[i] + 2];

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
                     = mm_force[3 * mmAtoms_.collectiveIndex()[i]];
        pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 1]
                     = mm_force[3 * mmAtoms_.collectiveIndex()[i] + 1];
        pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 2]
                     = mm_force[3 * mmAtoms_.collectiveIndex()[i] + 2];

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

    // Py_XDECREF(pFunc);
    // Py_XDECREF(pModule);
    // Py_XDECREF(pPDBname);
    // Py_XDECREF(pQMBasis);
    // Py_XDECREF(pQMMult);
    // Py_XDECREF(pysfcReturn);
    // Py_XDECREF(pQMCharge);
    // Py_XDECREF(pQMMMEnergy);
    // Py_XDECREF(pQMForce);
    // Py_XDECREF(pMMForce);
    fprintf(stderr, "test1!!!\n");
    // Py_XDECREF(npyQMForce);
    // Py_XDECREF(npyQMForce_rows);
    // Py_XDECREF(npyQMForce_columns);
    // Py_XDECREF(npyMMForce);
    // Py_XDECREF(npyMMForce_rows);
    // Py_XDECREF(npyMMForce_columns);
    fprintf(stderr, "test2!!!\n");
    Py_FinalizeEx();
    fprintf(stderr, "test3!!!\n");

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
    double Fxfout = 0.0, Fyfout = 0.0 , Fzfout = 0.0;
    double coordx = 0.0, coordy = 0.0, coordz = 0.0;
    double charge = 0.0;
    QMMM_record += formatString("%16.9f\n", c_hartreeBohr2Md);
    for (size_t i = 0; i < qmAtoms_.numAtomsLocal(); i++)
    {

        QMMM_record += formatString("%5s QM  ", periodic_system[parameters_.atomNumbers_[qmAtoms_.globalIndex()[i]]].c_str());
        Fxfout = fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][XX]/c_hartreeBohr2Md;
        Fyfout = fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][YY]/c_hartreeBohr2Md;
        Fzfout = fOutput->forceWithVirial_.force_[qmAtoms_.localIndex()[i]][ZZ]/c_hartreeBohr2Md;
        Fx = static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]]]);
        Fy = static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 1]);
        Fz = static_cast<real>(pyscfForce[3 * qmAtoms_.globalIndex()[qmAtoms_.collectiveIndex()[i]] + 2]);

        // Fx = Fx/c_bohr2ANG;
        // Fy = Fy/c_bohr2ANG;
        // Fz = Fz/c_bohr2ANG;

        coordx = fInput.x_[qmAtoms_.localIndex()[i]][XX];
        coordy = fInput.x_[qmAtoms_.localIndex()[i]][YY];
        coordz = fInput.x_[qmAtoms_.localIndex()[i]][ZZ];

        charge = fInput.chargeA_[qmAtoms_.localIndex()[i]];

        QMMM_record += formatString("%7.3lf %7.3lf %7.3lf", coordx, coordy, coordz);
        QMMM_record += formatString("%14.10lf %14.10lf %14.10lf %14.10lf %14.10lf %14.10lf", Fx, Fy, Fz, Fxfout, Fyfout, Fzfout);
        QMMM_record += formatString("%7.3f %4d %4d %4d %4d\n", 0.0, static_cast<int>(i), qmAtoms_.localIndex()[i], qmAtoms_.globalIndex()[i], qmAtoms_.collectiveIndex()[i]);
    }


    for (size_t i = 0; i < mmAtoms_.numAtomsLocal(); i++)
    {

        QMMM_record += formatString("%5s MM  ", periodic_system[parameters_.atomNumbers_[mmAtoms_.globalIndex()[i]]].c_str());
        Fxfout = fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][XX]/c_hartreeBohr2Md;
        Fyfout = fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][YY]/c_hartreeBohr2Md;
        Fzfout = fOutput->forceWithVirial_.force_[mmAtoms_.localIndex()[i]][ZZ]/c_hartreeBohr2Md;
        Fx = static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]]]);
        Fy = static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 1]);
        Fz = static_cast<real>(pyscfForce[3 * mmAtoms_.globalIndex()[mmAtoms_.collectiveIndex()[i]] + 2]);

        // Fx = Fx/c_bohr2ANG;
        // Fy = Fy/c_bohr2ANG;
        // Fz = Fz/c_bohr2ANG;

        coordx = fInput.x_[mmAtoms_.localIndex()[i]][XX];
        coordy = fInput.x_[mmAtoms_.localIndex()[i]][YY];
        coordz = fInput.x_[mmAtoms_.localIndex()[i]][ZZ];

        charge = fInput.chargeA_[mmAtoms_.localIndex()[i]];
        QMMM_record += formatString("%7.3lf %7.3lf %7.3lf", coordx, coordy, coordz);
        QMMM_record += formatString("%14.10lf %14.10lf %14.10lf %14.10lf %14.10lf %14.10lf", Fx, Fy, Fz, Fxfout, Fyfout, Fzfout);
        QMMM_record += formatString("%7.3f %4d %4d %4d %4d\n", charge, static_cast<int>(i), mmAtoms_.localIndex()[i], mmAtoms_.globalIndex()[i], mmAtoms_.collectiveIndex()[i]);
    }

    QMMM_record += formatString("\n");

    recordFile << QMMM_record;

    recordFile.close();

    return;
}


} // namespace gmx
