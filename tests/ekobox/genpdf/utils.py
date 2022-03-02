# -*- coding: utf-8 -*-
import os
import pathlib
from contextlib import contextmanager

test_pdf = pathlib.Path(__file__).parent / "genpdf"

# thanks https://stackoverflow.com/questions/431684/how-do-i-change-the-working-directory-in-python/24176022#24176022
@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
