import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i","--input_dir", type=str, required=True)
args = parser.parse_args()

input_dir=args.input_dir

for root,dirs,files in os.walk(input_dir):
    for file in files:
        if file=="no_vocals.wav":
            os.remove(os.path.join(root,file))