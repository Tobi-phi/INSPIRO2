import json
from pathlib import Path

# Function to get visualization parameters from visual.json
def get_visual():
    visual_path = Path(__file__).parent / 'visual.json'
    with open(visual_path, 'r') as f:
        visual = json.load(f)
    return visual

# Define paths for this file
data_path = Path(__file__).parent / 'data'
output_path = Path(__file__).parent / 'output'

# Define default jd start 
# As a star jd_start_default = 2460899 -> utc: 2025-08-01 12:00:00 is used
jd_start_default = 2460889