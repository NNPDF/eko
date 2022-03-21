# -*- coding: utf-8 -*-

import pytest

from ekobox import genpdf


def benchmark_genpdf_exceptions(tmp_path, cd):
    # using a wrong label and then a wrong parent pdf
    with cd(tmp_path):
        with pytest.raises(TypeError):
            genpdf.generate_pdf(
                "test_genpdf_exceptions1",
                ["f"],
                {
                    21: lambda x, Q2: 3.0 * x * (1.0 - x),
                    2: lambda x, Q2: 4.0 * x * (1.0 - x),
                },
            )
        with pytest.raises(ValueError):
            genpdf.generate_pdf(
                "test_genpdf_exceptions2",
                ["g"],
                10,
            )
        with pytest.raises(FileExistsError):
            genpdf.install_pdf("foo")

        with pytest.raises(TypeError):
            genpdf.generate_pdf("debug", [21], info_update=(10, 15, 20))
