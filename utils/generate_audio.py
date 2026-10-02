import os
import subprocess

def convert(v, output_path):
    out_file = os.path.join(output_path, os.path.basename(v)[:-4] + '.wav')
    subprocess.check_call([
        'ffmpeg',
        '-n',
        '-i', v,
        '-acodec', 'pcm_s16le',
        '-ac','1',
        '-ar','16000',
        out_file
    ])

folder_path = '/data/Kinetics-600/video/'
num = 0

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.mp4'):
            file_path = os.path.join(root, file)
            output_path = root
            os.makedirs(output_path, exist_ok=True)
            try:
                convert(file_path, output_path)
                num += 1
            except Exception as e:
                print("Failed:", file_path, e)

print(f"Created {num} .wav files")