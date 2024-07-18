"""Parse abbreviations from sphinx to Rust."""

import pathlib
import re

SRC = (
    pathlib.Path(__file__).parents[1]
    / "doc"
    / "source"
    / "shared"
    / "abbreviations.rst"
)

cnt = SRC.read_text("utf-8")
for el in cnt.split(".."):
    test = re.match(r"\s*(.+)\s+replace::\n\s+:abbr:`(.+?)\((.+)\)`", el.strip())
    if test is None:
        continue
    # Print to terminal - the user can dump to the relevant file
    print(f'"{test[2].strip()}": "{test[3].strip()}",')
