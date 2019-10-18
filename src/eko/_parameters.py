import numpy as np

# default in Python3 is np.float64,np.complex128
# so any raw number, such as 11./3., is given in 64bit and determines thus the precision
# keep in mind, that is most likely bottlenecked by the integration routines
t_float = np.float64
t_complex = np.complex128
