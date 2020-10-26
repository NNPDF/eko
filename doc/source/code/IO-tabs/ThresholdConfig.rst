ThresholdConfig parameters:

- :py:obj:`str` ``FNS`` active flavor number scheme. Possible values are `FFNS`,
  representing a |FFNS|, or, `ZM-VFNS`, `FONLL-A`, `FONLL-A'`,
  representing a |VFNS|; **Required**
- :py:obj:`float` ``Q0`` reference scale from which to start; **Required**
- :py:obj:`int` ``NfFF`` number of fixed flavors if applicable; **Required** if in |FFNS|, ignored otherwise
- :py:obj:`float` ``mc`` charm mass in GeV; **Required** if in |VFNS|, ignored otherwise
- :py:obj:`float` ``mb`` bottom mass in GeV; **Required** if in |VFNS|, ignored otherwise
- :py:obj:`float` ``mt`` top mass in GeV; **Required** if in |VFNS|, ignored otherwise