# -*- coding: utf-8 -*-
"""
    Convert raw PDF data of :cite:`Giele:2002hx` to Python
"""

if __name__ == "__main__":
    # helper to extract the numbers from the pdf
    raw = """"""
    for r in raw:
        r = (
            r.replace("∗\n", "")
            .replace(" −", "e-")
            .replace(" +", "e+")
            .replace("\n", ",")
        )
        print("[" + r + "]")