# -*- coding: utf-8 -*-

class DFdict(dict):
    """
    .. todo:: write the docs
    """

    def __init__(self, *args):
        super(DFdict, self).__init__(*args)
        self.msgs = []

    def print(self, *msgs, sep=" ", end="\n"):
        if len(msgs) > 0:
            self.msgs.append(msgs[0])

            for msg in msgs[1:]:
                self.msgs.append(sep)
                self.msgs.append(msg)
        self.msgs.append(end)

    def __setitem__(self, key, value):
        self.print(f"PID: {key}")
        self.print(value)
        self.print()
        super(DFdict,self).__setitem__(key,value)

    def __repr__(self):
        return "".join([str(x) for x in self.msgs])
