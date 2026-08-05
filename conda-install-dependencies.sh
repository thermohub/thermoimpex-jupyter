#!/bin/bash
# Installs the C++/Python dependencies needed to run the thermoimpex-jupyter
# notebooks: jsonarango, jsonio17, jsonimpex17, ChemicalFun, ThermoFun and
# ThermoHubClient, followed by the `thermomatch` Python module itself, built
# with its Qt GUI application disabled (this repo only drives it from Python,
# it never launches the ThermoMatch desktop app), and GEMSGUI's `json2db`
# CLI tool (JSON -> GEM-Selektor .pdb/.ndx converter, also GUI-disabled --
# see the json2db block below).
#
# This is a trimmed copy of thermomatch's own conda-install-dependencies.sh
# (https://bitbucket.org/gems4/thermomatch) with the GUI-only pieces removed:
# jsonui17 and thermofungui (both only used to build the Qt desktop app) are
# skipped, and qt6-charts/qt6-webengine are correspondingly absent from
# environment.devenv.yml (qt6-main IS present -- see the comment there).
#
# Needs gcc v.5 or higher, cmake, git and (for the ArangoDB-backed import
# steps) ArangoDB server installed locally -- see TECHINFO.md.
#
# Run this from inside an activated `thermoimpex` conda environment, i.e.
# after `conda devenv && conda activate thermoimpex` (see TECHINFO.md).

set -e

# CONDA_PREFIX is only exported by `conda activate` (local dev flow). Inside
# a repo2docker/Binder image build, `binder/postBuild` runs as a plain shell
# command with the env already on PATH but without CONDA_PREFIX set -- fall
# back to repo2docker's own NB_PYTHON_PREFIX, then to sys.prefix, before
# giving up.
if [ -z "${CONDA_PREFIX}" ]; then
    if [ -n "${NB_PYTHON_PREFIX}" ]; then
        export CONDA_PREFIX="${NB_PYTHON_PREFIX}"
    else
        PY="$(command -v python3 || command -v python)"
        if [ -n "${PY}" ]; then
            export CONDA_PREFIX="$("${PY}" -c 'import sys; print(sys.prefix)')"
        fi
    fi
fi

if [ -z "${CONDA_PREFIX}" ]; then
    echo "No active conda environment detected (CONDA_PREFIX is empty, and"
    echo "NB_PYTHON_PREFIX/python were not found to fall back to)."
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
# TODO: switch to master once the json2db tool lands there -- it currently
# only exists on this GEMSGUI branch.
BRANCH_GEMSGUI=fix_pow_other_bugs

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
        status=$?
        cd ~ && rm -rf ~/code
        [ $status -eq 0 ] || { echo "ERROR: jsonarango build/install failed" >&2; exit 1; }
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
        status=$?
        cd ~ && rm -rf ~/code
        [ $status -eq 0 ] || { echo "ERROR: jsonio17 build/install failed" >&2; exit 1; }
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
        status=$?
        cd ~ && rm -rf ~/code
        [ $status -eq 0 ] || { echo "ERROR: jsonimpex17 build/install failed" >&2; exit 1; }
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
        status=$?
        cd ~ && rm -rf ~/code
        [ $status -eq 0 ] || { echo "ERROR: eigen3 build/install failed" >&2; exit 1; }
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
        status=$?
        cd ~ && rm -rf ~/code
        [ $status -eq 0 ] || { echo "ERROR: pybind11 build/install failed" >&2; exit 1; }
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
        status=$?
        cd ~ && rm -rf ~/code
        [ $status -eq 0 ] || { echo "ERROR: ChemicalFun build/install failed" >&2; exit 1; }
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
        status=$?
        cd ~ && rm -rf ~/code
        [ $status -eq 0 ] || { echo "ERROR: ThermoFun build/install failed" >&2; exit 1; }
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
        status=$?
        cd ~ && rm -rf ~/code
        [ $status -eq 0 ] || { echo "ERROR: ThermoHubClient build/install failed" >&2; exit 1; }
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
        status=$?
        cd ~ && rm -rf ~/code
        [ $status -eq 0 ] || { echo "ERROR: thermomatch build/install failed" >&2; exit 1; }
}

# GEMSGUI's json2db tool (converts JSON database records into GEM-Selektor
# .pdb/.ndx files -- see databases/THEREDA/data-out for example input/output).
# We only want this one CLI tool, not the gem-selektor desktop app, so this
# is built with -DBUILD_APP=OFF -DBUILD_TOOLS=OFF -DBUILD_EXPORT=ON: that
# skips the GUI app/library entirely (no gem-selektor binary gets built) and
# only needs Qt6::Core (see GEMSGUI's top-level CMakeLists.txt), which is why
# qt6-main is in environment.devenv.yml but qt6-charts/qt6-webengine are not.
# `cmake --install` places the binary at ${CONDA_PREFIX}/bin/json2db and its
# required runtime data (schema/config, via -s) at
# ${CONDA_PREFIX}/share/gemsgui/Resources.
test -f ${CONDA_PREFIX}/bin/json2db || {
        mkdir -p ~/code && \
                cd ~/code && \
                git clone https://github.com/gemshub/GEMSGUI.git -b $BRANCH_GEMSGUI && \
                cd GEMSGUI && \
                mkdir -p build && \
                cd build && \
                cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_APP=OFF -DBUILD_TOOLS=OFF -DBUILD_EXPORT=ON -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} && \
                cmake --build . -j $threads --target json2db && \
                cmake --install . && \
                test -f ${CONDA_PREFIX}/bin/json2db
        status=$?
        cd ~ && rm -rf ~/code
        [ $status -eq 0 ] || { echo "ERROR: GEMSGUI json2db build/install failed" >&2; exit 1; }
}
# Pre-create json2db's -u (user profile) directory once, here, rather than
# per-call: TVisor::firstTimeSetup() (triggered whenever -u doesn't exist
# yet) does a real filesystem copy keyed off Resources/projects/, which we
# deliberately don't install (see benchcomp/CMakeLists.txt in GEMSGUI) to
# keep json2db's installed footprint ~400KB instead of Resources/'s full
# ~61MB. Pre-creating it here means that code path never runs at all.
mkdir -p ${CONDA_PREFIX}/var/gemsgui-user/projects

echo "Done. 'import thermomatch', 'import thermofun', 'import chemicalfun' and"
echo "'import thermohubclient' should now work from Python inside this conda env."
echo "'json2db' (JSON -> GEM-Selektor .pdb/.ndx) should now be on PATH; pass"
echo "'-s \${CONDA_PREFIX}/share/gemsgui' so it finds its Resources/ data."
