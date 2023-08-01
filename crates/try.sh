pip install -e ekuad

printf "\nTry to use the CFFI package:\n"
python -c "import ekuad; print(ekuad.lib.quad_ker(4, True))"
