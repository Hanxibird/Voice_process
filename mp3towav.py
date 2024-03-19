from pydub import AudioSegment
import os
import argparse
def convert_mp3_to_wav(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp3'):
                mp3_path = os.path.join(root, file)
                wav_path = os.path.splitext(mp3_path)[0] + '.wav'
                audio = AudioSegment.from_mp3(mp3_path)
                audio.export(wav_path, format='wav')
                print(f"Converted: {mp3_path} to {wav_path}")
                os.remove(mp3_path)
                print(f"Deleted: {mp3_path}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    args = parser.parse_args()
    convert_mp3_to_wav(args.input_dir)
