import eko


class MockRunner:
    def __init__(self, *_args):
        pass

    def get_output(self):
        return "output"


def test_run(monkeypatch):
    # just test, that it is a shortcut to get_output
    monkeypatch.setattr(eko.runner.legacy, "Runner", MockRunner)
    o = eko.solve({}, {})
    assert o == "output"
