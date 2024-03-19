import os
import sys
import glob
import subprocess

model_type = "htdemucs_ft"
input_dir = sys.argv[1]
demucs_out = sys.argv[2]
for dir in os.listdir(input_dir):
    for song_dir in glob.glob(os.path.join(os.path.join(input_dir,dir), '*')):
        command = ["python", "-m", "demucs", "--two-stems=vocals", "-n", model_type, song_dir, "-o", demucs_out]
        subprocess.run(command)