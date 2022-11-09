from eko import runcards as rc

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
