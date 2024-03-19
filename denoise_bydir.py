import denoise.AIDenoise as AIDenoise
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("-i","--input_dir", type=str, required=True)
parser.add_argument("-o","--output_dir", type=str, required=True)
args = parser.parse_args()
input_dir=args.input_dir
for root,dirs,files in os.walk(input_dir):
    for file in files:
        if file.endswith(".wav"):
            AIDenoise.process_in_chunks(os.path.join(root,file), os.path.join(args.output_dir,file))

