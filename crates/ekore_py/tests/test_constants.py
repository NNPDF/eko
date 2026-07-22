import ekore_py


def test_max_orders():
    assert ekore_py.constants.MAX_ORDER_QCD == 4
    assert ekore_py.constants.MAX_ORDER_QED == 2


def test_pids():
    assert ekore_py.constants.PID_NSP == 10101
    assert ekore_py.constants.PID_NSM == 10201
    assert ekore_py.constants.PID_NSV == 10200
    assert ekore_py.constants.PID_NSP_U == 10102
    assert ekore_py.constants.PID_NSP_D == 10103
    assert ekore_py.constants.PID_NSM_U == 10202
    assert ekore_py.constants.PID_NSM_D == 10203
