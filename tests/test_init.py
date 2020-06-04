# -*- coding: utf-8 -*-

import eko


class MockRunner:
    def __init__(self, *_args):
        pass

    def get_output(self):
        return "output"


def test_run():
    # just test, that it is a shortcut to get_output
    eko.runner.Runner = MockRunner
    o = eko.run_dglap({})
    assert o == "output"
