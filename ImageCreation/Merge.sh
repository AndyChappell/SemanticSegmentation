#!/bin/bash

for momentum in 1 3 5 7
do
    cd /r07/dune/sg568/LAr/Jobs/protoDUNE/2019/September/ProtoDUNE_HierarchyMetrics_DeepLearning_Training/AnalysisTag3/mcc11_Pndr/Beam_Cosmics/${momentum}GeV/NoSpaceCharge/Images

    nFiles=`ls -l | grep jpg | wc -l`
    ## 3 views, 4 rotations, input and truth images (24 total)
    # 3 views, input and truth images (6 total)
    nEvents="$(($nFiles / 6))"

    for jobNumber in $(seq 1 $nEvents)
    do
        #for rotation in 0 90 180 270
        for rotation in 0
        do
            for view in U V W
            do
                convert +append -set colorspace RGB InputImage_ProtoDUNE_HierarchyMetrics_DeepLearning_Training_Job_Number_${jobNumber}_CaloHitList${view}_${rotation}.jpg TruthImage_ProtoDUNE_HierarchyMetrics_DeepLearning_Training_Job_Number_${jobNumber}_CaloHitList${view}_${rotation}.jpg MergedImage_ProtoDUNE_HierarchyMetrics_DeepLearning_Training_Job_Number_${jobNumber}_CaloHitList${view}_${rotation}.jpg
            done
        done
    done
done
