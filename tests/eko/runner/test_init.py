import pathlib

import eko


class MockRunner:
    def __init__(self, theory, operator, path: pathlib.Path):
        self.path = path

    def compute(self):
        self.path.write_text("output", encoding="utf-8")


def test_run(monkeypatch, tmp_path: pathlib.Path):
    # just test, that it is a shortcut to 'compute'
    path = tmp_path / "eko.tar"
    monkeypatch.setattr(eko.runner.legacy, "Runner", MockRunner)
    eko.solve({}, {}, path=path)
    out = path.read_text(encoding="utf-8")
    assert out == "output"
