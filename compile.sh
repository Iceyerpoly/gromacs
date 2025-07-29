#!/bin/bash
#SBATCH --partition=shared
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=10000

source ~/.bashrc
conda activate Gromacs
module purge
 
export TMPDIR=/local/$USER/gromacs_compile
mkdir -p "$TMPDIR"
cd ../gromacs
rm -rf build
mkdir -p build && cd build
 
cmake .. -DBUILD_SHARED_LIBS=OFF -DGMXAPI=OFF -DGMX_INSTALL_NBLIB_API=OFF -DGMX_DOUBLE=ON -DGMX_BUILD_OWN_FFTW=ON -DGMX_PYSCF=ON
# make -j 8
make -j 8
# -DCMAKE_BUILD_TYPE=Debug
 
rm -rf $TMPDIR