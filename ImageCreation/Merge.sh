#!/bin/bash

if [[ ( "$1" == "" ) || ( "$2" == "" ) ]]; then
   echo -e "One or more arguments missing (./merge.sh <dir> <pattern>), using defaults:"
   echo -e "   dir = /storage/epp2/phrdqd/Pandora.git/Images"
   echo -e "   pattern = DUNEFD_MC11"
   dir="/storage/epp2/phrdqd/Pandora.git/Images"
   pattern="DUNEFD_MC11"
else
   dir=$1
   pattern=$2
   echo -e "dir = $dir"
   echo -e "patterm = $pattern"
fi

cd $dir

nFiles=`ls -l | grep jpg | wc -l`
# 3 views, input and truth images (6 total)
nEvents="$(($nFiles / 6))"
let "nEvents = nEvents - 1"

for job in $(seq 0 $nEvents)
do
   for view in U V W
   do
      infile=InputImage_${pattern}_CaloHitList${view}_${job}_0.jpg
      truthfile=TruthImage_${pattern}_CaloHitList${view}_${job}_0.jpg
      outfile=MergedImage_${pattern}_CaloHitList${view}_${job}_0.jpg
      convert +append -set colorspace RGB $infile $truthfile $outfile
   done
done

