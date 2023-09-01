import unittest
import numpy
import pyscf
import sys
from pyscf import lib, gto, scf, grad, dft
from pyscf.qmmm import itrf
import csv
from pyscf.data import elements, radii, nist

def QMcalculation(pPDBname, pQMBasis, pQMMult, pQMCharge):

    # pPDBname = "_cp2k.pdb"
    # pQMBasis = "ccpvdz"
    # pQMMult = 1
    # pQMCharge = 0
    qmatoms = []
    qmcoords = []
    qmkinds = []
    mmsymbols = []
    mmcoords = []
    mmcharges = []
    with open(pPDBname, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=" ", skipinitialspace=True)
        for line in csv_reader:
            if int(line[4]) == 1:
                qmatoms.append([line[-2], (float(line[5]), float(line[6]), float(line[7]))])
                qmkinds.append(line[-2])
                qmcoords.append((float(line[5]), float(line[6]), float(line[7])))
            else:
                mmsymbols.append(line[-2])
                mmcoords.append((float(line[5]), float(line[6]), float(line[7])))
                mmcharges.append(float(line[-1]))
        # print(qmcoords)
        # print(qmkinds)
        # print(qmatoms)

    # tmp = mmsymbols[0]
    # print(tmp, type(tmp))
    # tmp_charge = elements.charge(tmp)
    # print(tmp_charge)

    mmatmns = []
    mmradii = []
    # print("mmsymbols", mmsymbols, type(mmsymbols), "\n")
    for tmp in mmsymbols:
         mmatmn_tmp = elements.charge(tmp)
         mmatmns.append(mmatmn_tmp)
         mmradii.append(radii.COVALENT[mmatmn_tmp]*nist.BOHR)
    # print(mmatmns)
    # print(mmradii)

    # print(radii.COVALENT[1])

    mol = gto.M(atom=qmatoms, unit='ANG', basis=pQMBasis)
    mf_qm = dft.RKS(mol)
    mf_qm.xc = 'PBE'
    mf = itrf.mm_charge(mf_qm, mmcoords, mmcharges, mmradii, unit='ANG')

    energy = mf.kernel()
    # print("energy = ", energy, type(energy))

    # print("mmcharges\n", mmcharges, type(mmcharges))
    # print("mmcoords\n", mmcoords, type(mmcoords))
    dm = mf.make_rdm1()
    dm_shape = dm.shape
    # print("\ndensity matrix", dm_shape, "\n", dm, type(dm))

    j3c = mol.intor('int1e_grids', hermi=1, grids=numpy.asarray(mmcoords)/0.529177211)
    j3c_shape = j3c.shape
    # print("\nj3c", j3c_shape, "\n", j3c, type(j3c))
    j3c_sum = numpy.einsum('kpq,k->pq', j3c, -numpy.asarray(mmcharges))
    j3c_sum_shape = j3c_sum.shape
    # print("\nj3c_sum", j3c_sum_shape, "\n", j3c_sum, type(j3c_sum))

    energy_qme_mm = numpy.einsum('pq,qp->',j3c_sum,dm)
    # print("h1e_mm_energy = ", energy_qme_mm, type(energy_qme_mm))

    energy_qmnuc_mm = 0
    # print(mol.natm)
    for i in range(mol.natm):
        for j in range(len(mmcoords)):
            qm_q, qm_r = mol.atom_charge(i), mol.atom_coord(i)
            # print("qm_q = ", qm_q, "qm_r = ", qm_r)
            mm_q, mm_r = numpy.asarray(mmcharges[j]), numpy.asarray(mmcoords[j])/0.529177211
            # print("mm_q = ", mm_q, "mm_r = ", mm_r)
            r = lib.norm(qm_r-mm_r)
            energy_qmnuc_mm += qm_q*(mm_q/r)
    # print("qmenergy_qmnuc_mm = ", energy_qmnuc_mm, type(energy_qmnuc_mm))
    qmmm_energy = energy_qmnuc_mm + energy_qme_mm
    # print("qmmm_energy = ", qmmm_energy)

    # energy_hcore = mf.get_hcore()
    # print("get_hcore = ", energy_hcore, type(energy_hcore))
    # energy_qmnuc_mm = mf.energy_nuc()
    # print("qmnuc_mm_energy = ", energy_qmnuc_mm, type(energy_qmnuc_mm))
    # print("qm_mm_energy = ", energy_qm_mm, type(energy_qm_mm))

    mf_grad = itrf.mm_charge_grad(grad.RKS(mf), mmcoords, mmcharges, mmradii, unit='ANG')
    qmforce = -mf_grad.kernel()
    # print("qmforce", type(qmforce))
    # print(qmforce)

    mmforce_qmnuc = -mf_grad.grad_nuc_mm()
    mmforce_qme = -mf_grad.grad_hcore_mm(dm)

    # print("mmforce_qmnuc", type(mmforce_qmnuc))
    # print(mmforce_qmnuc)
    # print("mmforce_e", type(mmforce_qme))
    # print(mmforce_qme)

    mmforce = numpy.add(mmforce_qmnuc,mmforce_qme)
    # print("mmforce", type(mmforce))
    # print(mmforce)

    # print("reference count of pPDBname is ",sys.getrefcount(pPDBname))

    return energy, qmforce, mmforce

def qmmmCalc(qmbasis, qmmult, qmcharge, qmkinds, qmcoords, mmkinds, mmcharges, mmcoords):

    # pPDBname = "_cp2k.pdb"
    # pQMBasis = "ccpvdz"
    # pQMMult = 1
    # pQMCharge = 0
    # qmatoms = []
    # qmcoords = []
    # qmkinds = []
    # mmsymbols = []
    # mmcoords = []
    # mmcharges = []
    # with open(pPDBname, 'r') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=" ", skipinitialspace=True)
    #     for line in csv_reader:
    #         if int(line[4]) == 1:
    #             qmatoms.append([line[-2], (float(line[5]), float(line[6]), float(line[7]))])
    #             qmkinds.append(line[-2])
    #             qmcoords.append((float(line[5]), float(line[6]), float(line[7])))
    #         else:
    #             mmsymbols.append(line[-2])
    #             mmcoords.append((float(line[5]), float(line[6]), float(line[7])))
    #             mmcharges.append(float(line[-1]))
        # print(qmcoords)
        # print(qmkinds)


    qmatoms = []
    for kind, coord in zip(qmkinds, qmcoords):
        qmatom_tmp = [kind] + coord
        qmatoms.append(qmatom_tmp)
    # print(qmatoms)
    # tmp = mmsymbols[0]
    # print(tmp, type(tmp))
    # tmp_charge = elements.charge(tmp)
    # print(tmp_charge)

    mmatmns = []
    mmradii = []
    # print("mmsymbols", mmsymbols, type(mmsymbols), "\n")
    for tmp in mmkinds:
         mmatmn_tmp = elements.charge(tmp)
         mmatmns.append(mmatmn_tmp)
         mmradii.append(radii.COVALENT[mmatmn_tmp]*nist.BOHR)
    # print(mmatmns)
    # print(mmradii)

    # print(radii.COVALENT[1])

    mol = gto.M(atom=qmatoms, unit='ANG', basis=qmbasis)
    mf_qm = dft.RKS(mol)
    mf_qm.xc = 'PBE'
    mf = itrf.mm_charge(mf_qm, mmcoords, mmcharges, mmradii, unit='ANG')

    energy = mf.kernel()
    # print("energy = ", energy, type(energy))

    # print("mmcharges\n", mmcharges, type(mmcharges))
    # print("mmcoords\n", mmcoords, type(mmcoords))
    dm = mf.make_rdm1()
    dm_shape = dm.shape
    # print("\ndensity matrix", dm_shape, "\n", dm, type(dm))

    j3c = mol.intor('int1e_grids', hermi=1, grids=numpy.asarray(mmcoords)/0.529177211)
    j3c_shape = j3c.shape
    # print("\nj3c", j3c_shape, "\n", j3c, type(j3c))
    j3c_sum = numpy.einsum('kpq,k->pq', j3c, -numpy.asarray(mmcharges))
    j3c_sum_shape = j3c_sum.shape
    # print("\nj3c_sum", j3c_sum_shape, "\n", j3c_sum, type(j3c_sum))

    energy_qme_mm = numpy.einsum('pq,qp->',j3c_sum,dm)
    # print("h1e_mm_energy = ", energy_qme_mm, type(energy_qme_mm))

    energy_qmnuc_mm = 0
    # print(mol.natm)
    for i in range(mol.natm):
        for j in range(len(mmcoords)):
            qm_q, qm_r = mol.atom_charge(i), mol.atom_coord(i)
            # print("qm_q = ", qm_q, "qm_r = ", qm_r)
            mm_q, mm_r = numpy.asarray(mmcharges[j]), numpy.asarray(mmcoords[j])/0.529177211
            # print("mm_q = ", mm_q, "mm_r = ", mm_r)
            r = lib.norm(qm_r-mm_r)
            energy_qmnuc_mm += qm_q*(mm_q/r)
    # print("qmenergy_qmnuc_mm = ", energy_qmnuc_mm, type(energy_qmnuc_mm))
    qmmm_energy = energy_qmnuc_mm + energy_qme_mm
    # print("qmmm_energy = ", qmmm_energy)

    # energy_hcore = mf.get_hcore()
    # print("get_hcore = ", energy_hcore, type(energy_hcore))
    # energy_qmnuc_mm = mf.energy_nuc()
    # print("qmnuc_mm_energy = ", energy_qmnuc_mm, type(energy_qmnuc_mm))
    # print("qm_mm_energy = ", energy_qm_mm, type(energy_qm_mm))

    mf_grad = itrf.mm_charge_grad(grad.RKS(mf), mmcoords, mmcharges, mmradii, unit='ANG')
    qmforce = -mf_grad.kernel()
    # print("qmforce", type(qmforce))
    # print(qmforce)

    mmforce_qmnuc = -mf_grad.grad_nuc_mm()
    mmforce_qme = -mf_grad.grad_hcore_mm(dm)

    # print("mmforce_qmnuc", type(mmforce_qmnuc))
    # print(mmforce_qmnuc)
    # print("mmforce_e", type(mmforce_qme))
    # print(mmforce_qme)

    mmforce = numpy.add(mmforce_qmnuc,mmforce_qme)
    # print("mmforce", type(mmforce))
    # print(mmforce)

    # print("reference count of pPDBname is ",sys.getrefcount(pPDBname))

    return energy, qmforce, mmforce

def printProp(prop):
    print(prop)

# QMcalculation("_cp2k_HFHOH.pdb", "631g*", 1, 0)
