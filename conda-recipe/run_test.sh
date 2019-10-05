#!/bin/bash
set -u
set -v
set -e

#Python tests for the installed validphys package
pytest --pyargs eko

#Print linkage data
conda inspect linkages -p $PREFIX $PKG_NAME
