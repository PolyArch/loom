#!/usr/bin/env python3

import os
import pathlib
import sys


def main() -> int:
    root = pathlib.Path(sys.argv[1])
    ddi_root = pathlib.Path(sys.argv[2])
    root.mkdir(parents=True, exist_ok=True)
    (ddi_root / "bin").mkdir(parents=True, exist_ok=True)
    executables = (
        "dc_shell",
        "fc_shell",
        "genus",
        "innovus",
        "joules",
        "openroad",
        "pt_shell",
        "quartus_sh",
        "tempus",
        "vcs",
        "vivado",
        "voltus",
        "xrun",
        "yosys",
    )
    for executable in executables:
        path = root / executable
        path.write_text("#!/usr/bin/bash\necho wrong-version\n", encoding="ascii")
        path.chmod(0o755)
    verilator = root / "verilator"
    verilator.write_text(
        "#!/usr/bin/bash\necho 'Verilator 5.050 fixture'\n", encoding="ascii"
    )
    verilator.chmod(0o755)
    genus = ddi_root / "bin" / "genus"
    genus.write_text(
        "#!/usr/bin/bash\n"
        "echo 'Program Name: Genus(TM) Synthesis Solution, Version: 26.10-p002_1'\n",
        encoding="ascii",
    )
    genus.chmod(0o755)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
