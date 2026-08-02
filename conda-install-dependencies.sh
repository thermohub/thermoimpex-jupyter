#!/bin/bash
# Installs the C++/Python dependencies needed to run the thermoimpex-jupyter
# notebooks: jsonarango, jsonio17, jsonimpex17, ChemicalFun, ThermoFun and
# ThermoHubClient, followed by the `thermomatch` Python module itself, built
# with its Qt GUI application disabled (this repo only drives it from Python,
# it never launches the ThermoMatch desktop app).
#
# This is a trimmed copy of thermomatch's own conda-install-dependencies.sh
# (https://bitbucket.org/gems4/thermomatch) with the GUI-only pieces removed:
# jsonui17 and thermofungui (both only used to build the Qt desktop app) are
# skipped, and qt6-main/qt6-charts/qt6-webengine are correspondingly absent
# from environment.devenv.yml.
#
# Needs gcc v.5 or higher, cmake, git and (for the ArangoDB-backed import
# steps) ArangoDB server installed locally -- see TECHINFO.md.
#
# Run this from inside an activated `thermoimpex` conda environment, i.e.
# after `conda devenv && conda activate thermoimpex` (see TECHINFO.md).

set -e

if [ -z "${CONDA_PREFIX}" ]; then
    echo "No active conda environment detected (CONDA_PREFIX is empty)."
    echo "Run 'conda devenv && conda activate thermoimpex' first."
    exit 1
fi

if [ "$(uname)" == "Darwin" ]; then
    EXTN=dylib
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    EXTN=so
fi

threads=${THREADS:-10}
BRANCH_JSON=master
BRANCH_TFUN=master
BRANCH_THERMOMATCH=master

echo "conda prefix: ${CONDA_PREFIX}"

# jsonArango database client (velocypack from ArangoDB)
test -f ${CONDA_PREFIX}/lib/libjsonarango.$EXTN || {
        mkdir -p ~/code && \
                cd ~/code && \
                git clone --recurse-submodules https://bitbucket.org/gems4/jsonarango.git && \
                cd jsonarango && \
                mkdir -p build && \
                cd build && \
                cmake .. -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DBULID_LOCAL_TESTS=OFF -DBULID_REMOTE_TESTS=OFF -DCMAKE_INSTALL_RPATH=${CONDA_PREFIX}/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE && \
                make -j $threads && \
                make install
        cd ~ && rm -rf ~/code
}

# JSONIO17 database client
test -f ${CONDA_PREFIX}/lib/libjsonio17.$EXTN || {
        mkdir -p ~/code && \
                cd ~/code && \
                git clone https://bitbucket.org/gems4/jsonio17.git -b $BRANCH_JSON && \
                cd jsonio17 && \
                mkdir -p build && \
                cd build && \
                cmake .. -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DBuildTests=OFF -DCMAKE_INSTALL_RPATH=${CONDA_PREFIX}/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE && \
                make -j $threads && \
                make install
        cd ~ && rm -rf ~/code
}

# JSONIMPEX (needed to build the thermomatch python module)
test -f ${CONDA_PREFIX}/lib/libjsonimpex17.$EXTN || {
        mkdir -p ~/code && \
                cd ~/code && \
                git clone https://bitbucket.org/gems4/jsonimpex17.git -b $BRANCH_JSON && \
                cd jsonimpex17 && \
                mkdir -p build && \
                cd build && \
                cmake .. -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DBuildTests=OFF -DCMAKE_INSTALL_RPATH=${CONDA_PREFIX}/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE && \
                make -j $threads && \
                make install
        cd ~ && rm -rf ~/code
}

# Eigen3 math library
test -d ${CONDA_PREFIX}/include/eigen3/Eigen || {
        mkdir -p ~/code && \
                cd ~/code && \
                git clone https://gitlab.com/libeigen/eigen.git -b 3.4.0 && \
                cd eigen && \
                mkdir -p build && \
                cd build && \
                cmake .. && \
                make install
        cd ~ && rm -rf ~/code
}

# pybind11
test -d ${CONDA_PREFIX}/include/pybind11 || {
        mkdir -p ~/code && \
                cd ~/code && \
                git clone https://github.com/pybind/pybind11.git && \
                cd pybind11 && \
                mkdir -p build && \
                cd build && \
                cmake .. -DPYBIND11_TEST=OFF && \
                make && \
                make install
        cd ~ && rm -rf ~/code
}

# ChemicalFun library
test -f ${CONDA_PREFIX}/lib/libChemicalFun.$EXTN || {
        mkdir -p ~/code && \
                cd ~/code && \
                git clone https://github.com/thermohub/chemicalfun -b $BRANCH_TFUN && \
                cd chemicalfun && \
                mkdir -p build && \
                cd build && \
                cmake .. -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DCHEMICALFUN_BUILD_PYTHON=OFF && \
                make -j $threads && \
                make install
        cd ~ && rm -rf ~/code
}

# ThermoFun library
test -f ${CONDA_PREFIX}/lib/libThermoFun.$EXTN || {
        mkdir -p ~/code && \
                cd ~/code && \
                git clone https://github.com/thermohub/thermofun -b $BRANCH_TFUN && \
                cd thermofun && \
                mkdir -p build && \
                cd build && \
                cmake .. -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DTFUN_BUILD_PYTHON=OFF -DCMAKE_PREFIX_PATH=${CONDA_PREFIX}/include/spdlog/ && \
                make -j $threads && \
                make install
        cd ~ && rm -rf ~/code
}

# ThermoHubClient library
test -f ${CONDA_PREFIX}/lib/libThermoHubClient.$EXTN || {
        mkdir -p ~/code && \
                cd ~/code && \
                git clone https://bitbucket.org/gems4/thermohubclient.git -b $BRANCH_TFUN && \
                cd thermohubclient && \
                mkdir -p build && \
                cd build && \
                cmake .. -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_RPATH=${CONDA_PREFIX}/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE && \
                make -j $threads && \
                make install
        cd ~ && rm -rf ~/code
}

# thermomatch itself (Python module only -- Qt GUI application disabled)
test -f ${CONDA_PREFIX}/lib/python*/site-packages/thermomatch/PyThermoMatch*.$EXTN || {
        mkdir -p ~/code && \
                cd ~/code && \
                git clone https://bitbucket.org/gems4/thermomatch.git -b $BRANCH_THERMOMATCH && \
                cd thermomatch && \
                mkdir -p build && \
                cd build && \
                cmake .. -DCMAKE_BUILD_TYPE=Release -DTHERMOMATCH_APPLICATION=OFF -DTHERMOMATCH_BUILD_PYTHON=ON -DCMAKE_INSTALL_RPATH=${CONDA_PREFIX}/lib -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE && \
                make -j $threads && \
                make install
        cd ~ && rm -rf ~/code
}

echo "Done. 'import thermomatch', 'import thermofun', 'import chemicalfun' and"
echo "'import thermohubclient' should now work from Python inside this conda env."
