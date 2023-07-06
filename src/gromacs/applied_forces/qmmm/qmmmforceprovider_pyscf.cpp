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
#include <iostream>

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

void QMMMForceProvider::initPyscfForceEnvironment(const t_commrec& cr)
{

} // namespace gmx

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

    std::cout << "test!!!" << std:endl;

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
        if (cr.nnodes > 1)
        {
        }
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
};

} // namespace gmx
