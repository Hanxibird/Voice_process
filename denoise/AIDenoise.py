import numpy as np
import librosa
import torch
import os
from torch.autograd import Variable
from hyperparams_ns import Hyperparams as hp1
import model_deploy_ns
import soundfile as sf
import sys,os
sys.path.append(os.getcwd())
import argparse
class AIDenoise:
    def __init__(self):
        path = os.path.abspath(os.path.dirname(__file__))
        ckpt_newest_ns = os.path.join(path, 'model_epoch_state_dic_1487000_-5.6104')
        params_ns = torch.load(ckpt_newest_ns)
        self.net_ns = model_deploy_ns.Net_v6_80bark()
        self.net_ns.load_state_dict(params_ns)
        self.net_ns.cuda()
        self.bin2bark_matrix_ns = np.array(np.load(os.path.join(path, "bin2bark_48000_1025_80.npy")), dtype=np.float32)
        self.bark2bin_matrix_ns = np.array(np.load(os.path.join(path, "bark2bin_48000_1025_80.npy")), dtype=np.float32)

    def get_stft_matrix(self, magnitudes, phases):
        return magnitudes * np.exp(1.j * phases)

    def to_wav_file(self, mag, phase, len_hop=hp1.hop_length):
        stft_maxrix = self.get_stft_matrix(mag, phase)
        return np.array(
            librosa.istft(stft_maxrix, hop_length=len_hop, win_length=hp1.win_length, window=np.hamming(hp1.win_length),
                          center=True))

    def ai_denoise(self, audio_data):
        theata = 0.00
        stft = librosa.stft(audio_data, n_fft=hp1.n_fft, hop_length=hp1.hop_length, win_length=hp1.win_length,
                            window=np.hamming(hp1.win_length), center=True)
        output_stft_mag = np.zeros((stft.shape[0], stft.shape[1]))
        stft_mag = np.transpose(np.abs(stft), [1, 0])
        stft_mag = np.pad(stft_mag, ((hp1.reciep_filed, 0), (0, 0)), 'constant')
        stft_bark = np.dot(np.power(stft_mag, 2), self.bin2bark_matrix_ns)
        stft_mag = np.expand_dims(stft_mag, axis=0)
        stft_mag = np.expand_dims(stft_mag, axis=0)

        stft_bark = np.expand_dims(stft_bark, axis=0)
        stft_bark = np.expand_dims(stft_bark, axis=0)

        input = torch.from_numpy(stft_bark).cuda()

        # x = Variable(input).cuda()
        x = Variable(input)

        output = self.net_ns(x).detach().cpu().numpy()
        # for i in range(output.shape[2]):
        #    np.savetxt('./output_torch004.model/output_%d.txt'%i, output[0, 0, i, :], delimiter=',')
        output = np.dot(output[:, :, :, :80], self.bark2bin_matrix_ns)
        output = np.minimum(output, 1)

        output_stft_mag = np.transpose(
            (stft_mag[0, 0, hp1.reciep_filed:, :] * (output[0, 0, :, :] * (1 - theata) + theata)), [1, 0])
        stft_phase = np.angle(stft)

        wav_clean = self.to_wav_file(output_stft_mag, stft_phase)

        return wav_clean

    def process(self, audio_dir):
        audio_data, sr = librosa.load(audio_dir, sr=hp1.sr, mono=False)
        if audio_data.shape[0] == 2:
            audio_data = audio_data[0, :]
        audio_ns = self.ai_denoise(audio_data)
        print(f"{audio_dir}: AI denoise done!")

        return audio_ns
def process_in_chunks(input_file, output_file, chunk_duration=60, sample_rate=48000):
    # Initialize the denoising model
    ai_denoise = AIDenoise()

    # Read the input audio file
    audio, sr = librosa.load(input_file, sr=hp1.sr, mono=False)
    if audio.shape[0] == 2:
        audio = audio[0, :]
    # Calculate the number of samples per chunk
    chunk_size = chunk_duration * sample_rate

    # Process and write the audio in chunks
    with sf.SoundFile(output_file, mode='w', samplerate=sr, channels=1) as out_file:
        for start in range(0, len(audio), chunk_size):
            end = min(start + chunk_size, len(audio))
            audio_chunk = audio[start:end]
            denoised_chunk = ai_denoise.ai_denoise(audio_chunk)
            out_file.write(denoised_chunk)
            print("done")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    process_in_chunks(args.input, "denoised.wav")




