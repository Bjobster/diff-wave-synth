"""
Preprocess NSynth dataset to extract pitch and loudness.
"""
import librosa
import numpy as np
from core import extract_loudness, extract_pitch
from multiprocessing.dummy import Pool as ThreadPool
import yaml 
import os
from tqdm import tqdm
import tensorflow as tf

# if you find problems using TF, disable the GPU and inspect
# tf.config.set_visible_devices([], 'GPU')


def preprocess(f, sampling_rate, block_size, signal_length, oneshot, 
               target_pitch_file, target_loudness_file, pitch_model_capacity="full", **kwargs):
    # print(f"Processing file: {f}") 
    if os.path.exists(os.path.join(target_pitch_file, os.path.basename(f).replace(".wav", "_pitch.npy"))):
        print(f"Skipping file: {f}") 
        # print("Skipping...")
    
    else:
        try:
            x, sr = librosa.load(path=f, sr=sampling_rate)
            # print(f"File {f} loaded with length {len(x)} and sampling rate {sr}")
        except Exception as e:
            print(f"Error loading file {f}: {e}")
            return

        N = (signal_length - len(x) % signal_length) % signal_length
        x = np.pad(x, (0, N))

        if oneshot:
            x = x[..., :signal_length]

        # v2 is based on my own version of torchcrepe, comment out for now
        # pitch = extract_pitch_v2(x, sampling_rate, block_size)
        
        # pitch = extract_pitch(x, sampling_rate, block_size, model_capacity=pitch_model_capacity)
        # loudness = extract_loudness(x, sampling_rate, block_size)
        try:
            pitch = extract_pitch(x, sampling_rate, block_size, model_capacity=pitch_model_capacity)
            loudness = extract_loudness(x, sampling_rate, block_size)
        except Exception as e:
            print(f"Error processing file {f}: {e}")
            return

        x = x.reshape(-1, signal_length)
        pitch = pitch.reshape(x.shape[0], -1).squeeze()
        loudness = loudness.reshape(x.shape[0], -1)

        try:
            np.save(os.path.join(target_pitch_file, os.path.basename(f).replace(".wav", "_pitch.npy")),pitch.squeeze())
            np.save(os.path.join(target_loudness_file, os.path.basename(f).replace(".wav", "_loudness.npy")),loudness.squeeze())
        except Exception as e:
            print(f"Error saving file {f}: {e}")
            return

        return x, pitch, loudness


if __name__ == "__main__":
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    
    asyncc = True
    paths = ["train_path", "valid_path", "test_path"]
    
    for path in paths:
        audio_path_prefix = os.path.join(config["dataset"][path], config["dataset"]["audio"])
        print(f"Audio path prefix: {audio_path_prefix}")
        audio_path = sorted(os.listdir(audio_path_prefix))
        print(f"Number of audio files in {audio_path_prefix}: {len(audio_path)}")
        print('Length: ', len(audio_path))

        if not os.path.exists(os.path.join(config["dataset"][path], config["dataset"]["pitch"])):
            print(f"Creating directories for path: {path}")
            os.mkdir(os.path.join(config["dataset"][path], config["dataset"]["pitch"]))
            os.mkdir(os.path.join(config["dataset"][path], config["dataset"]["loudness"]))

        if asyncc:
            pool = ThreadPool(4)
            pbar = tqdm(total=len(audio_path))

            def update(*a):
                pbar.update()
                
            def error_handler(e):
                print(f"Error in processing: {e}")
                                    
            for i in range(pbar.total):
                # print(f"Submitting file to thread pool: {audio_path[i]}") 
                pool.apply_async(preprocess, 
                                args=(os.path.join(audio_path_prefix, audio_path[i]),
                                    config["common"]["sampling_rate"], 
                                    config["common"]["block_size"], 
                                    config["common"]["sampling_rate"] * config["common"]["duration_secs"], 
                                    True,
                                    os.path.join(config["dataset"][path], config["dataset"]["pitch"]),
                                    os.path.join(config["dataset"][path], config["dataset"]["loudness"])), 
                                callback=update,
                                error_callback=error_handler)
            pool.close()
            pool.join()
        
        else:
            for i in tqdm(range(len(audio_path))):
                preprocess(os.path.join(audio_path_prefix, audio_path[i]), 
                            config["common"]["sampling_rate"], 
                            config["common"]["block_size"], 
                            config["common"]["sampling_rate"] * config["common"]["duration_secs"], 
                            True,
                            os.path.join(config["dataset"][path], config["dataset"]["pitch"]),
                            os.path.join(config["dataset"][path], config["dataset"]["loudness"]),
                            pitch_model_capacity=config["crepe"]["model"])