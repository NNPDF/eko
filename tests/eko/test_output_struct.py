# -*- coding: utf-8 -*-
from eko import output
from eko.output import struct


class TestLegacy:
    def test_items(self, fake_output):
        """Test autodump, autoload, and manual unload."""
        eko = output.EKO.new(theory={}, operator=fake_output)
        for q2, op in fake_output["Q2grid"].items():
            eko[q2] = output.Operator.from_dict(op)

        q2 = next(iter(fake_output["Q2grid"]))

        eko._operators[q2] = None
        assert isinstance(eko[q2], struct.Operator)
        assert isinstance(eko._operators[q2], struct.Operator)

        del eko[q2]

        assert eko._operators[q2] is None

    def test_iter(self, fake_output):
        """Test managed iteration."""
        eko = output.EKO.new(theory={}, operator=fake_output)
        for q2, op in fake_output["Q2grid"].items():
            eko[q2] = output.Operator.from_dict(op)

        q2prev = None
        for q2, op in eko:
            if q2prev is not None:
                assert eko._operators[q2prev] is None
            assert isinstance(op, struct.Operator)
            q2prev = q2

    def test_context_operator(self, fake_output):
        """Test automated handling through context."""
        eko = output.EKO.new(theory={}, operator=fake_output)
        for q2, op in fake_output["Q2grid"].items():
            eko[q2] = output.Operator.from_dict(op)

        q2 = next(iter(fake_output["Q2grid"]))

        with eko.operator(q2) as op:
            assert isinstance(op, struct.Operator)

        assert eko._operators[q2] is None
