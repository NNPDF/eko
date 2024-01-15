#!/bin/bash

SCHEME="FFNS"
SV="central"
AD_VAR=(0 0 0 0 0 0 0)
AD_MAX_VAR_LIST=(19 21 15 6)

# run the central
python run-n3lo.py $SCHEME $SV "${AD_VAR[@]}"

# loop on gammas
for I in {0..3}
    do
    # loop on variations
    VARIATIONS=${AD_MAX_VAR_LIST[$I]}
    VAR_LIST=($(seq 1 1 $VARIATIONS))
    for VAR in "${VAR_LIST[@]}"
        do
        AD_VAR[$I]=$VAR
        python run-n3lo.py $SCHEME $SV "${AD_VAR[@]}"
        AD_VAR[$I]=0
        done
    done
