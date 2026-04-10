"""LHAPDF data parser."""

import pathlib
from dataclasses import dataclass
from io import StringIO

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class LhapdfDataBlock:
    """LHAPDF data block."""

    xgrid: npt.NDArray[np.float64]
    """X grid."""

    qgrid: npt.NDArray[np.float64]
    """Q grid."""

    pids: npt.NDArray[np.int_]
    """|PID| grid."""

    data: npt.NDArray[np.float64]
    """Tabulated data."""

    def is_valid(self) -> bool:
        """Check if dimensions are reasonable."""
        for a in [self.xgrid, self.qgrid, self.pids]:
            if len(a.shape) != 1 or len(a) <= 0:
                return False
        return self.data.shape == (len(self.xgrid) * len(self.qgrid), len(self.pids))

    def add(self, other):
        """Add other block."""
        # x and q have to be the same
        np.testing.assert_allclose(self.xgrid, other.xgrid)
        np.testing.assert_allclose(self.qgrid, other.qgrid)
        # PID we can recover
        tot_pids = np.unique(np.concatenate([self.pids, other.pids]))
        tot_data = []
        for pid in tot_pids:
            if pid in self.pids and pid in other.pids:
                tot_data.append(
                    self.data[:, self.pids.searchsorted(pid)].copy()
                    + other.data[:, other.pids.searchsorted(pid)].copy()
                )
            elif pid in self.pids:
                tot_data.append(self.data[:, self.pids.searchsorted(pid)].copy())
            elif pid in other.pids:
                tot_data.append(other.data[:, other.pids.searchsorted(pid)].copy())
        return LhapdfDataBlock(
            xgrid=self.xgrid, qgrid=self.qgrid, pids=tot_pids, data=np.array(tot_data).T
        )


class LhapdfDataFile:
    """LHAPDF data file."""

    header: dict[str, str]
    """Header information."""

    blocks: list[LhapdfDataBlock]
    """Data blocks."""

    def __init__(self, header: dict[str, str], blocks: list[LhapdfDataBlock]) -> None:
        self.header = header
        self.blocks = blocks

    @classmethod
    def read(cls, path: pathlib.Path):
        """Read from file."""
        cnt = path.read_text().split("---\n")
        # header
        header = {}
        for line in cnt[0].splitlines():
            k, v = line.split(":", 1)
            header[k] = v.strip()
        # blocks
        blocks = []
        for b in cnt[1:]:
            # skip last if necessary
            if len(b.strip()) == 0:
                continue
            # subdivide
            lines = b.splitlines()
            xgrid = np.fromstring(lines[0], np.float64, sep=" ")
            qgrid = np.fromstring(lines[1], np.float64, sep=" ")
            pids = np.fromstring(lines[2], np.int_, sep=" ")
            data = np.loadtxt(StringIO("\n".join(lines[3:])))
            blocks.append(
                LhapdfDataBlock(xgrid=xgrid, qgrid=qgrid, pids=pids, data=data)
            )
        return cls(header=header, blocks=blocks)

    def write(self, path: pathlib.Path) -> int:
        """Write to file."""
        cnt = ""
        # header
        for k, v in self.header.items():
            cnt += f"{k}: {v}\n"
        if "Format" not in self.header:
            cnt += "Format: lhagrid1\n"
        cnt += "---\n"
        # blocks
        blocks = []
        for b in self.blocks:
            bcnt = StringIO()
            np.savetxt(bcnt, b.xgrid, "%.8e", newline=" ")
            bcnt.write("\n")
            np.savetxt(bcnt, b.qgrid, "%.8e", newline=" ")
            bcnt.write("\n")
            np.savetxt(bcnt, b.pids, "%d", newline=" ")
            bcnt.write("\n")
            np.savetxt(bcnt, b.data, "%.8e")
            blocks.append(bcnt.getvalue())
        cnt += "---\n".join(blocks)
        cnt += "---\n"
        return path.write_text(cnt)

    @classmethod
    def read_with_set(cls, path: pathlib.Path, pdfset: str, member: int = 0):
        """Read given `member` from given `pdfset` inside `path`."""
        return cls.read(path / pdfset / f"{pdfset}_{member:04d}.dat")
