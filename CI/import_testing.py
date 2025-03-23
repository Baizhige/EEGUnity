import json
import os
import sys
# Get the parent directory of the script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from eegunity import UnifiedDataset
from eegunity import get_data_row
from eegunity import Pipeline


u_ds = UnifiedDataset("/gpfs/work/int/chengxuanqin21/science_works/EEGUnity_CI/bcic_iv_2a/")
