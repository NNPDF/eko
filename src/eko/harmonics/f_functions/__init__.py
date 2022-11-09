r"""This subpackage contains some Mellin transformations for 5th order harmonics sum.

Notation refers to:

    - :cite:`Blumlein:2009ta`. Johannes Blumlein. Structural Relations of
      Harmonic Sums and Mellin Transforms up to Weight w = 5. Comput. Phys.
      Commun., 180:2218-2249, 2009. arXiv:0901.3106,
      doi:10.1016/j.cpc.2009.07.004.

Mellin transform is defined with the convention x^(N).
F19, F20, F21 are not present explicitly in the paper
"""  # pylint: disable=line-too-long

from .f9 import F9
from .f11 import F11
from .f13 import F13
from .f14_f12 import F14F12
from .f16 import F16
from .f17 import F17
from .f18 import F18
from .f19 import F19
from .f20 import F20
from .f21 import F21
