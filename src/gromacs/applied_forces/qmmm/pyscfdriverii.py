import unittest
import numpy
import pyscf
from pyscf import lib, gto, scf, grad, dft, neo
from pyscf.qmmm import itrf
from pyscf.data import elements, radii, nist
from pyscf.qmmm.mm_mole import create_mm_mol

def qmmmCalc(qmbasis, qmmult, qmcharge,
             qmindex, qmkinds, qmcoords,
             mmindex, mmkinds, mmcharges, mmcoords, links):

    # include the link atoms in qm-related lists
    [qmkinds_link, qmcoords_link, qmindex_link] = link_coord_corr(links, qmindex, qmkinds, qmcoords, mmindex, mmkinds, mmcoords)
    # # print("qmcharge", qmcharge)
    print("qmindex_link", qmindex_link)
    # print("qmkinds_link", qmkinds_link)
    # print("qmcoords_link", qmcoords_link)

    # print("mmindex", mmindex)
    # print("mmkinds", mmkinds)
    # print("mmcharges", mmcharges)
    # print("mmcoords", mmcoords)
    # print("links", links)
    qmatoms = []
    for kind, coord in zip(qmkinds_link, qmcoords_link):
        qmatom = [kind] + coord
        qmatoms.append(qmatom)

    coords_gen_xyz("qm coords modified", qmindex_link, qmkinds_link, qmcoords_link)
    coords_gen_xyz("mm coords", mmindex, mmkinds, mmcoords)

    # qmbasis = 'def2-SVP'
    mmradii = []

# set the mmhost charge to 0
    for i in range(len(mmindex)):
        # print(i, len(mmindex))
        if i == len(mmindex):
            break
        for j in range(len(links)):
            if mmindex[i] == links[j][1]:
                mmcharges[i] = 0
        print("%5d %5s %5f" % (mmindex[i], mmkinds[i], mmcharges[i]))


    for kind in mmkinds:
        charge = elements.charge(kind)
        mmradii.append(radii.COVALENT[charge]*nist.BOHR)
        # mmradii.append(1e-8*nist.BOHR)

    mol = gto.M(atom=qmatoms, unit='ANG', basis='6-31G*', charge=qmcharge)
    mf_dft = dft.RKS(mol)
    mf_dft.xc = 'BLYP'
    # mf_dft = mf_dft.density_fit(auxbasis='def2-svp-ri')
    mf = itrf.mm_charge(mf_dft, mmcoords, mmcharges, mmradii)
    # mf = itrf.mm_charge(mf_dft, mmcoords, mmcharges)

    energy = mf.kernel()
    dm = mf.make_rdm1()

    mf_grad = itrf.mm_charge_grad(grad.RKS(mf), mmcoords, mmcharges, mmradii)
    qmforces = -mf_grad.kernel()

    mmforces_qmnuc = -mf_grad.grad_nuc_mm()
    mmforces_qme = -mf_grad.grad_hcore_mm(dm)

    mmforces = mmforces_qmnuc + mmforces_qme
    # print("mmforces", type(mmforces))
    # print("mmforces", type(qmforces))
    # print("energy", type(energy))
    coords_gen_xyz("qm forces",qmindex_link, qmkinds_link, qmforces)
    coords_gen_xyz("mm forces",mmindex, mmkinds, mmforces)
    [qmindex_force_corr, qmforces_link, mmforces_link] = link_force_corr(links, qmindex_link, qmkinds_link, qmforces, mmindex, mmkinds, mmforces)
    coords_gen_xyz("qm forces modified", qmindex_force_corr, qmkinds_link, qmforces_link)
    coords_gen_xyz("mm forces modified", mmindex, mmkinds, mmforces_link)

    return energy, qmforces_link, mmforces_link
    return qmcharge, qmcoords, mmcoords

def printProp(prop):
    print(prop)

def coords_gen_xyz(xyzpropname, index, kinds, xyzprops):
    print("--------------------", xyzpropname, "--------------------")
    for i in range(len(xyzprops)):
        print("%4s %4s %18.12f %18.12f %18.12f" % (index[i], kinds[i], xyzprops[i][0], xyzprops[i][1], xyzprops[i][2]))

def link_coord_corr(links, qmindex, qmkinds, qmcoords, mmindex, mmkinds, mmcoords):
    print('qm index', qmindex)
    print('mm index', mmindex)
    # print(qmcoords)
    print("there are", len(links), "links to be processed")
    for i in range((len(links))):
        qmindex.append('L'+str(i))
        # print(i+1,"th link is between [QM, MM] =", links[i])
        # print('link''s qm host collective index', qmindex.index(links[i][0]))
        # print('link''s mm host collective index', mmindex.index(links[i][1]))
        # link_coord = [0.000, 0.000, 0.000]
        qm_c_index = qmindex.index(links[i][0])
        mm_c_index = mmindex.index(links[i][1])
        qm_host_coord = numpy.array(qmcoords[qm_c_index])
        mm_host_coord = numpy.array(mmcoords[mm_c_index])
        alpha = 1.38 # need to enable user defined alpha(scaling factor) later
        alpha_rec = 1/alpha
        link_coord = alpha_rec * mm_host_coord + (1 - alpha_rec) * qm_host_coord
        link_coord = list(link_coord)
        # print("qm host atom is", qmkinds[qm_c_index], "coordinate:", qm_host_coord, type(qm_host_coord))
        # print("mm host atom is", mmkinds[qm_c_index], "coordinate:", mm_host_coord, type(mm_host_coord))

        # print("mm host atom is", 'H', "coordinate:", link_coord, type(link_coord))

        qmkinds.append('H  ') # need to enable user defined link atom kind later
        qmcoords.append(link_coord)
    # print(qmkinds)
    # print(qmcoords)

    return qmkinds, qmcoords, qmindex

def link_force_corr(links, qmindex, qmkinds, qmforces, mmindex, mmkinds, mmforces):

    linkcount = 0
    for i in range((len(links))):
        linkcount = linkcount + 1
        linkindex = 'L'+str(i)

        qm_c_index = qmindex.index(links[i][0])
        mm_c_index = mmindex.index(links[i][1])
        link_c_index = qmindex.index(linkindex)

        print(linkcount,"th link is between [QM, MM] =", links[i])
        print('link''s qm host collective index', qm_c_index)
        print('link''s mm host collective index', mm_c_index)
        print('link''s collective index', link_c_index)

        alpha = 1.38
        alpha_rec = 1/alpha

        link_force = qmforces[link_c_index]
        qm_link_force_partition = link_force * (1 - alpha_rec)
        mm_link_force_partition = link_force * alpha_rec
        qmforces[qm_c_index] += qm_link_force_partition
        mmforces[mm_c_index] += mm_link_force_partition
        print("link atom: kind", qmkinds[link_c_index], ", force: ", link_force, ", paritions: qm:", qm_link_force_partition,", mm:", mm_link_force_partition)

    for i in range((len(links))):

        linkindex = 'L'+str(i)
        link_c_index = qmindex.index(linkindex)
        # print(link_c_index, type(link_c_index))
        qmindex.remove(linkindex)
        qmforces = numpy.delete(qmforces, link_c_index, 0)
    # print(qmindex)
    # print(qmforces)

    return qmindex, qmforces, mmforces

if __name__ == '__main__':
    # qmbasis = '631G'
    # qmmult = 1
    # qmcharge = 0
    # qmkinds = ['O','H', 'H']
    # qmcoords = [[-1.464, 0.099, 0.300],
    #            [-1.956, 0.624, -0.340],
    #            [-1.797, -0.799, 0.206]]
    # mmkinds = ['O','H','H']
    # mmcharges = [-1.040, 0.520, 0.520]
    # # mmradii = [0.63, 0.32, 0.32]
    # mmcoords = [(1.369, 0.146,-0.395),
    #              (1.894, 0.486, 0.335),
    #              (0.451, 0.165,-0.083)]

    qmbasis = '631G'
    qmmult = 1

    # qmcharge = 0
    # qmindex = [4, 5, 13, 14, 15]
    # qmkinds = ['C  ', 'O  ', 'H  ', 'H  ', 'H  ']
    # qmcoords = [[46.97, 49.78, 50.5], [45.55, 49.81, 50.5], [47.33, 48.94, 51.04], [47.23, 49.73, 49.459999999999994], [45.21, 49.86, 51.8]]
    # mmindex = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12]
    # mmkinds = ['C  ', 'O  ', 'C  ', 'C  ', 'H  ', 'H  ', 'H  ', 'H  ', 'H  ', 'H  ', 'H  ']
    # mmcharges = [-0.0521, -0.3809, 0.0113, -0.2261, 0.0824, 0.0824, 0.0824, 0.0767, 0.0767, 0.1073, 0.1073]
    # mmcoords = [[51.0, 50.98, 50.010000000000005], [49.6, 51.01, 49.97], [49.059999999999995, 50.99, 51.31999999999999], [47.53, 51.0, 51.22], [51.28999999999999, 51.06, 48.96], [51.44, 50.15, 50.58], [51.33, 51.849999999999994, 50.53], [49.3, 51.900000000000006, 51.92], [49.400000000000006, 50.12, 51.849999999999994], [47.199999999999996, 51.91, 50.739999999999995], [47.04, 51.07, 52.26]]
    # links = [[4, 3]]

    qmcharge = 0
    qmindex = [0, 5, 6, 7, 8, 15]
    qmkinds = ['C  ', 'O  ', 'H  ', 'H  ', 'H  ', 'H  ']
    qmcoords = [[51.0, 50.98, 50.010000000000005], [45.55, 49.81, 50.5], [51.28999999999999, 51.06, 48.96], [51.44, 50.15, 50.58], [51.33, 51.849999999999994, 50.53], [45.21, 49.86, 51.8]]
    mmindex = [1, 2, 3, 4, 9, 10, 11, 12, 13, 14]
    mmkinds = ['O  ', 'C  ', 'C  ', 'C  ', 'H  ', 'H  ', 'H  ', 'H  ', 'H  ', 'H  ']
    mmcharges = [-0.3809, 0.0113, -0.2261, 0.0124, 0.0767, 0.0767, 0.1073, 0.1073, 0.1034, 0.1034]
    mmcoords = [[49.6, 51.01, 49.97], [49.059999999999995, 50.99, 51.31999999999999], [47.53, 51.0, 51.22], [46.97, 49.78, 50.5], [49.3, 51.900000000000006, 51.92], [49.400000000000006, 50.12, 51.849999999999994], [47.199999999999996, 51.91, 50.739999999999995], [47.04, 51.07, 52.26], [47.33, 48.94, 51.04], [47.23, 49.73, 49.459999999999994]]
    links = [[0, 1], [5, 4]]

    # output = link_coord_corr(links, qmindex, qmkinds, qmcoords, mmindex, mmkinds, mmcoords)
    # print(output[0], output[1])
    [energy, qmforce, mmforce] = qmmmCalc(qmbasis, qmmult, qmcharge,
             qmindex, qmkinds, qmcoords,
             mmindex, mmkinds, mmcharges, mmcoords, links)
    # qmmmCalcNEO(qmbasis, qmmult, qmcharge, qmkinds, qmcoords, mmkinds, mmcharges, mmcoords)
