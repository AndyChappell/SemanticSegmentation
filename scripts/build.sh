#!/bin/bash
export PATH_TO_TORCH="/cvmfs/uboone.opensciencegrid.org/products/libtorch/v1_0_1/Linux64bit+3.10-2.17-e17-prof/lib/python2.7/site-packages/torch"
export MY_TEST_AREA=`pwd`
export LD_LIBRARY_PATH="${PATH_TO_TORCH}/lib/":${LD_LIBRARY_PATH}
git clone https://github.com/PandoraPFA/PandoraPFA.git
git clone https://github.com/PandoraPFA/LArReco.git
git clone https://github.com/PandoraPFA/LArMachineLearningData.git
cd PandoraPFA
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="${PATH_TO_TORCH}/share/cmake" -DPANDORA_MONITORING=ON -DPANDORA_EXAMPLE_CONTENT=OFF -DPANDORA_LAR_CONTENT=ON -DPANDORA_LC_CONTENT=OFF -DCMAKE_CXX_FLAGS="-std=c++17 -Wno-implicit-fallthrough" ..
make -j4 install
cd $MY_TEST_AREA
cd LArReco
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="${PATH_TO_TORCH}/share/cmake" -DCMAKE_MODULE_PATH="$MY_TEST_AREA/PandoraPFA/cmakemodules" -DPANDORA_MONITORING=ON -DPandoraSDK_DIR=$MY_TEST_AREA/PandoraPFA/ -DPandoraMonitoring_DIR=$MY_TEST_AREA/PandoraPFA/ -DLArContent_DIR=$MY_TEST_AREA/PandoraPFA/ -DCMAKE_CXX_FLAGS="-std=c++17" ..
make -j4 install
cd $MY_TEST_AREA
cd LArMachineLearningData
export FW_SEARCH_PATH=$FW_SEARCH_PATH:`pwd`
cd ../LArReco/settings
export FW_SEARCH_PATH=$FW_SEARCH_PATH:`pwd`
cd $MY_TEST_AREA

