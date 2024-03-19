import os
import sys
import glob
import subprocess

model_type = "htdemucs_ft"
input_dir = sys.argv[1]
demucs_out = sys.argv[2]

for song_dir in glob.glob(os.path.join(input_dir, '*')):
    command = ["python", "-m", "demucs", "--two-stems=vocals", "-n", model_type, song_dir, "-o", demucs_out]
    subprocess.run(command)