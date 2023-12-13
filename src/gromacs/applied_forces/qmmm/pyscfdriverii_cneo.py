import unittest
import numpy
import pyscf
from pyscf import lib, gto, scf, grad, dft, neo
from pyscf.qmmm import itrf
from pyscf.data import elements, radii, nist
from pyscf.qmmm.mm_mole import create_mm_mol

def qmmmCalc(qmbasis, qmmult, qmcharge, qmkinds, qmcoords, mmkinds, mmcharges, mmcoords):

    qmatoms = []
    for kind, coord in zip(qmkinds, qmcoords):
        qmatom = [kind] + coord
        qmatoms.append(qmatom)
    # print(qmatoms)

    qmbasis = 'def2-SVP'
    mmatmns = []
    mmradii = []
    for kind in mmkinds:
         charge = elements.charge(kind)
         mmatmns.append(charge)
         mmradii.append(radii.COVALENT[charge]*nist.BOHR) # TODO: double check this

    # print("qmkinds", qmkinds)
    # print("qmcoords", qmcoords)
    # print("mmkinds", mmkinds)
    # print("mmcharges", mmcharges)
    # print("mmcoords", mmcoords)
    # print(qmcharge)
    mmmol = create_mm_mol(mmcoords, mmcharges, mmradii)

    mol_neo = neo.M(atom=qmatoms, basis=qmbasis, nuc_basis='pb4d',
                quantum_nuc=['H'], mm_mol=mmmol,charge=qmcharge)

    # energy
    mf = neo.CDFT(mol_neo)
    mf.mf_elec.xc = 'B3LYP'
    energy = mf.kernel()

    # gradient
    # qm gradient
    g = mf.Gradients()
    g_qm = g.grad()
    # mm gradient
    g_mm = g.grad_mm()

    qmforce = - g_qm
    mmforce = - g_mm
    # print(mmforce)

    return energy, qmforce, mmforce

def printProp(prop):
    print(prop)

if __name__ == '__main__':
    qmbasis = 'ccpvdz'
    # qmbasis = '631G'
    qmmult = 1
    qmcharge = 0
    qmkinds = ['O','H', 'H']
    qmcoords = [[-1.464, 0.099, 0.300],
               [-1.956, 0.624, -0.340],
               [-1.797, -0.799, 0.206]]
    mmkinds = ['O','H','H']
    mmcharges = [-1.040, 0.520, 0.520]
    mmradii = [0.63, 0.32, 0.32]
    mmcoords = [(1.369, 0.146,-0.395),
                 (1.894, 0.486, 0.335),
                 (0.451, 0.165,-0.083)]

    qmmmCalc(qmbasis, qmmult, qmcharge, qmkinds, qmcoords, mmkinds, mmcharges, mmcoords)
