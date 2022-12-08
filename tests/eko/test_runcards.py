from eko.io import runcards as rc

fake_theory = dict(order=(2, 0))

fake_operator = dict(xgrid=[1e-3, 1e-2, 1e-1, 1.0])


class TestTheory:
    def test_init(self):
        t = rc.TheoryCard(order=fake_theory["order"])

        assert t.order == fake_theory["order"]

    def test_load(self):
        t0 = rc.TheoryCard(order=fake_theory["order"])
        t1 = rc.TheoryCard.load(fake_theory)

        assert t0 == t1


class TestOperator:
    def test_init(self):
        o = rc.OperatorCard(xgrid=fake_operator["xgrid"])

        assert o.xgrid == fake_operator["xgrid"]

    def test_load(self):
        o0 = rc.OperatorCard(xgrid=fake_operator["xgrid"])
        o1 = rc.OperatorCard.load(fake_operator)

        assert o0 == o1


theory1 = {"alphas": 0.1180, "alphaqed": 0.007496, "PTO": 2, "QED": 0, "Q0": 1.0}


def test_compatibility():
    new_theory, _ = rc.update(theory1, {})

    assert new_theory.order[0] == theory1["PTO"] + 1


def operator_dict(xgrid, pids):
    return dict(
        ev_op_max_order=2,
        ev_op_iterations=1,
        interpolation_xgrid=xgrid,
        interpolation_polynomial_degree=4,
        interpolation_is_log=True,
        backward_inversion=None,
        n_integration_cores=1,
        debug_skip_singlet=False,
        debug_skip_non_singlet=False,
        inputgrid=xgrid,
        targetgrid=xgrid,
        pids=pids,
        inputpids=pids,
        targetpids=pids,
    )


def test_compatibility_operators():
    xgrid = [1e-3, 1e-2, 1e-1, 1.0]
    pids = [21, -1, 1]

    _, new_operator = rc.update(theory1, operator_dict(xgrid, pids))

    assert new_operator is not None
    assert not isinstance(new_operator.configs.ev_op_max_order, int)
