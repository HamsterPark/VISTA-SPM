import os
import sys

# Cap per-process thread pools before numpy/torch load. Episodes run in parallel, and oversubscribed
# BLAS/OpenMP pools (torch keeps one thread per core busy during VIGIL inference) starved the
# simulator's synchronous scan render past the controller's 5 s reply timeout. An explicit setting wins.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

from .cli import main  # noqa: E402

sys.exit(main())
