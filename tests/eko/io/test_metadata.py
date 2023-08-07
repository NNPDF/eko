import logging

import pytest
import yaml

from eko.io import bases, metadata, paths


def test_metadata(tmp_path, caplog):
    m = metadata.Metadata((1.0, 3), bases.Bases([0.1, 1.0]))
    # errors
    with caplog.at_level(logging.INFO):
        m.update()
    assert "no file" in caplog.text
    with pytest.raises(RuntimeError):
        m.path
    # now modify
    m.path = tmp_path
    m.update()
    p = paths.InternalPaths(tmp_path)
    assert tmp_path.exists()
    assert p.metadata.exists()
    assert p.metadata.is_file()
    assert "version" in p.metadata.read_text()
    # change version
    m.version = "0.0.0-a1~really1.0.0"
    m.update()
    # if I read back the thing should be what I set
    mn = metadata.Metadata((1.0, 3), bases.Bases([0.1, 1.0]))
    mm = metadata.Metadata.load(tmp_path)
    assert m.path == tmp_path
    assert mm.version != mn.version
    assert mm.version == "0.0.0-a1~really1.0.0"
