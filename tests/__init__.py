import os
import sys

import neuralprocesses.coders.setconv as _setconv

# Let `einsum` fail if there is no optimised implementation available.
_setconv._einsum_allowed_fallback = False

# Add package to path.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, "..")))
