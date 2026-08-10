import sounddevice as sd
import soundfile as sf
import numpy as np
import os

SOURCE_DIR = './to record'
DEST_DIR = './recorded'
SAMPLE_RATE = 44100

PRE_ROLL_SECONDS = 0.5

if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

files_to_process = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.wav')]

for filename in files_to_process:
    source_path = os.path.join(SOURCE_DIR, filename)
    dest_path = os.path.join(DEST_DIR, filename)

    print(f"Processing {filename}")

    original_audio, sr = sf.read(source_path, dtype='float32')

    pre_roll_samples = int(PRE_ROLL_SECONDS * SAMPLE_RATE)
    pre_roll_silence = np.zeros(pre_roll_samples, dtype='float32')
    audio_to_play = np.concatenate([pre_roll_silence, original_audio])

    print("Recording.")
    recorded_audio = sd.playrec(audio_to_play, samplerate=SAMPLE_RATE, channels=1)

    sd.wait()
    print("Recording finished.")

    aligned_audio = recorded_audio[pre_roll_samples:]

    sf.write(dest_path, aligned_audio, SAMPLE_RATE)

print("All files processed!")