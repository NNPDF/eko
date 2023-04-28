#!/bin/bash

MODE=2
FILES='split.F trans_mat_elems.F'

if [ ! -d 'N3LO_additions' ]; then
    git clone https://github.com/MSHTPDF/N3LO_additions
fi

cd 'N3LO_additions'

if [ "$MODE" -eq "1" ];
then
  f2py3 --overwrite-signature -m msht_n3lo -h ../msth_n3lo.template.pyf $FILES \
  only: pgg3a pgq3a pqg3a pqqps3a p3nsa p3nsb p3nsc ahg3a ahq3a aqqh3ab agqh3a aggh3a
fi;

if [ "$MODE" -eq "2" ];
 then
  f2py3 --f77flags="-ffixed-line-length-none -ffree-line-length-none" -m msht_n3lo -c ../msth_n3lo.template.pyf  $FILES \
  --fcompiler=gnu95
fi;
