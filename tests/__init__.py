import os
import sys

import lab as B

# Ease tests by adding a relatively big ridge.
B.epsilon = 1e-6

# Add package to path.
file_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(file_dir, "..")))
