#!/bin/bash

MODE=2
FILES='split.F'

# move back to root
# cd ..

if [ "$MODE" -eq "1" ];
then
  f2py3 --overwrite-signature -m msht_n3lo -h msth_n3lo.template1.pyf $FILES \
  only: pgg3a pgq3a  pqg3a  pqqps3a p3nsa p3nsb p3nsc
fi;

if [ "$MODE" -eq "2" ];
 then
  f2py3 --f77flags="-ffixed-line-length-none -ffree-line-length-none" -m msht_n3lo -c msth_n3lo.template.pyf  $FILES \
  -L/Users/giacomomagni/.conda/envs/eko_dev/lib --fcompiler=gnu95 
fi;

