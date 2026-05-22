from ekobox import genpdf


def test_load_info(fake_ct14):
    info = genpdf.load.load_info_from_file(fake_ct14)
    assert "SetDesc" in info
    assert "fake" in info["SetDesc"]
    assert sorted(info["Flavors"]) == sorted([-3, -2, -1, 21, 1, 2, 3])


def test_load_data_ct14(fake_ct14):
    f = genpdf.load.load_blocks_from_file(fake_ct14, 0)
    assert len(f.blocks) == 1
