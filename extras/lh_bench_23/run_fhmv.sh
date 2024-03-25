#!/bin/bash

SCHEME="VFNS"
SV="up"
AD_VAR=(0 0 0 0 0 0 0)


# run the central
python run-n3lo.py $SCHEME $SV "${AD_VAR[@]}" "--use_fhmruvv"

# loop on gammas
for I in {0..6}
    do
    # loop on variations
    for VAR in {1..2}
        do
        AD_VAR[$I]=$VAR
        python run-n3lo.py $SCHEME $SV "${AD_VAR[@]}" "--use_fhmruvv"
        AD_VAR[$I]=0
        done
    done
