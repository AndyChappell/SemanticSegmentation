source /cvmfs/uboone.opensciencegrid.org/products/setup_uboone.sh
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh

export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=softpipe
export DRAW_USE_LLVM=0

setup gcc v7_3_0 -f Linux64bit+3.10-2.17
setup git v2_4_6 -f Linux64bit+3.10-2.17
setup python v2_7_13d -f Linux64bit+3.10-2.17

setup eigen v3_3_3
setup root v6_16_00 -f Linux64bit+3.10-2.17 -q e17:prof
setup libtorch v1_0_1 -f Linux64bit+3.10-2.17 -q e17:prof
setup cmake v3_14_3 -f Linux64bit+2.6-2.12

#export LD_LIBRARY_PATH=/usera/marshall/Test/github/lib/:$LD_LIBRARY_PATH

#setup root v5_34_32 -f Linux64bit+3.10-2.17 -q e9:nu:prof
#source /usera/marshall/Test/root_v06_06_04/bin/thisroot.sh

#export PATH=/usera/marshall/Test/github/bin/:$PATH
#export LIBGL_ALWAYS_INDIRECT=1 #For visualisation within nx
