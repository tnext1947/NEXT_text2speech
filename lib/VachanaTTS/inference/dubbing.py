import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from transformers import pipeline
import scipy
from pathlib import Path
import srt
import moviepy.editor as mp
import numpy as np
from scipy.signal import resample_poly
import subprocess

# Output directory setup
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

def get_model_names(model_dir):
    model_paths = Path(model_dir).glob('*')
    return [model_path.name for model_path in model_paths if model_path.is_dir()]

def generate_speech(text, model_path):
    synthesiser = pipeline("text-to-speech", model=model_path, device=0 if torch.cuda.is_available() else -1)
    speech = synthesiser(text)
    
    resampled_audio = resample_poly(speech["audio"][0], 48000, speech["sampling_rate"])
    sampling_rate = 48000
    
    return sampling_rate, resampled_audio

def save_audio(sampling_rate, audio_data, filename="output.wav"):
    audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_data)
    return filename

def voice_cloning(speaker_files, reference_speakers, model_version, device_choice, vad_select):
    try:
        ckpt_converter = f'./OPENVOICE_MODELS/{model_version}'
        device = "cuda:0" if device_choice == "GPU" and torch.cuda.is_available() else "cpu"
        
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        
        cloned_files = {}
        
        for speaker_id, base_audio in speaker_files.items():
            if (speaker_id >= len(reference_speakers)) or (reference_speakers[speaker_id] is None):
                continue
                
            source_se, _ = se_extractor.get_se(base_audio, tone_color_converter, vad=vad_select)
            target_se, _ = se_extractor.get_se(reference_speakers[speaker_id], tone_color_converter, vad=vad_select)
            
            output_path = f'{output_dir}/speaker_{speaker_id + 1}_cloned.wav'
            tone_color_converter.convert(
                audio_src_path=base_audio,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path
            )
            cloned_files[speaker_id] = output_path
        
        if cloned_files:
            max_length = 0
            sample_rate = None
            audio_data = {}
            
            for speaker_id, file_path in cloned_files.items():
                sr, audio = scipy.io.wavfile.read(file_path)
                audio = audio.astype(np.float32) / 32767.0
                audio_data[speaker_id] = audio
                max_length = max(max_length, len(audio))
                sample_rate = sr
            
            combined = np.zeros(max_length, dtype=np.float32)
            for speaker_id, audio in audio_data.items():
                if np.max(np.abs(audio)) > 0:
                    normalized_audio = audio / np.max(np.abs(audio))
                    padded_audio = np.pad(normalized_audio, (0, max_length - len(audio)))
                    combined += padded_audio
            
            if np.max(np.abs(combined)) > 0:
                combined = combined * 0.9 / np.max(np.abs(combined))
                combined = np.int16(combined * 32767)
            
            final_path = f'{output_dir}/final_dubbed.wav'
            scipy.io.wavfile.write(final_path, sample_rate, combined)
            return final_path, "Voice cloning successful!"
        
        return None, "No voices to clone"
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, f"Error: {str(e)}"

def process_srt(srt_file):
    with open(srt_file, 'r', encoding='utf-8') as f:
        subtitles = list(srt.parse(f.read()))
    return subtitles

def generate_speech_from_srt(srt_file, model_paths, total_duration):
    subtitles = process_srt(srt_file)
    sampling_rate = 48000
    
    speaker_audio_files = {}
    speaker_timelines = {}
    
    for speaker_id in range(len(model_paths[1])):
        speaker_timelines[speaker_id] = np.zeros(int(total_duration * sampling_rate))
    
    for sub in subtitles:
        text = sub.content.strip()
        if text:
            try:
                if ',' in text:
                    speaker_id_str, text = text.split(',', 1)
                    speaker_id = int(speaker_id_str.strip()) - 1
                else:
                    speaker_id = 0
            except ValueError:
                speaker_id = 0
            
            if 0 <= speaker_id < len(model_paths[1]):
                model_path = os.path.join(model_paths[0], model_paths[1][speaker_id])
                _, audio_data = generate_speech(text, model_path)
                
                start_sample = int(sub.start.total_seconds() * sampling_rate)
                end_sample = int(sub.end.total_seconds() * sampling_rate)
                subtitle_duration = end_sample - start_sample
                
                if len(audio_data) > subtitle_duration:
                    audio_data = audio_data[:subtitle_duration]
                else:
                    audio_data = np.pad(audio_data, (0, subtitle_duration - len(audio_data)))
                
                audio_data = audio_data[:subtitle_duration]
                
                speaker_timelines[speaker_id][start_sample:end_sample] += audio_data
    
    for speaker_id, timeline in speaker_timelines.items():
        if np.any(timeline):
            speaker_file = f"{output_dir}/speaker_{speaker_id + 1}_base.wav"
            save_audio(sampling_rate, timeline, speaker_file)
            speaker_audio_files[speaker_id] = speaker_file
    
    return speaker_audio_files, sampling_rate

def dub_srt(srt_file, media_file, model_dir, model_names, reference_speakers, model_version, device_choice, vad_select, output_type, original_volume, dubbing_volume, clone_voice):
    if media_file.endswith(('.mp4', '.mkv', '.avi')):
        media = mp.VideoFileClip(media_file)
    else:
        media = mp.AudioFileClip(media_file)
    total_duration = media.duration
    
    speaker_files, sampling_rate = generate_speech_from_srt(
        srt_file, 
        (model_dir, model_names), 
        total_duration
    )
    
    if clone_voice:
        cloned_audio_file, status = voice_cloning(
            speaker_files,
            reference_speakers,
            model_version,
            device_choice,
            vad_select
        )
        
        if cloned_audio_file is None:
            return None, status
        
        dubbed_audio = mp.AudioFileClip(cloned_audio_file)
    else:
        combined_audio_path = f"{output_dir}/combined_speech.wav"
        combined_audio = np.zeros(int(total_duration * sampling_rate), dtype=np.float32)
        
        for speaker_id, audio_file in speaker_files.items():
            sr, audio = scipy.io.wavfile.read(audio_file)
            audio = audio.astype(np.float32) / 32767.0
            combined_audio[:len(audio)] += audio
        
        combined_audio = np.int16(combined_audio / np.max(np.abs(combined_audio)) * 32767)
        scipy.io.wavfile.write(combined_audio_path, sampling_rate, combined_audio)
        dubbed_audio = mp.AudioFileClip(combined_audio_path)
    
    dubbed_audio = dubbed_audio.volumex(dubbing_volume)
    
    if output_type == "Video" and media_file.endswith(('.mp4', '.mkv', '.avi')):
        original_audio = media.audio.volumex(original_volume)
        final_audio = mp.CompositeAudioClip([original_audio, dubbed_audio])
        final_video_path = f"{output_dir}/dubbed_video.mp4"
        
        if os.path.exists(final_video_path):
            os.remove(final_video_path)
        
        ffmpeg_command = [
            "ffmpeg",
            "-i", media_file,
            "-i", combined_audio_path if not clone_voice else cloned_audio_file,
            "-filter_complex", f"[0:a]volume={original_volume}[a0];[1:a]volume={dubbing_volume}[a1];[a0][a1]amix=inputs=2:duration=shortest",
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            final_video_path
        ]

        subprocess.run(ffmpeg_command, check=True)

        return final_video_path, "Dubbing completed!"
    else:
        final_audio_path = f"{output_dir}/dubbed_audio.wav"
        dubbed_audio.write_audiofile(final_audio_path)
        return final_audio_path, "Dubbing completed!"

if __name__ == "__main__":
    model_dir = "./models_mms"
    model_names = get_model_names(model_dir)
    max_speakers = 4

    srt_file = "path/to/srt_file.srt"
    media_file = "path/to/media_file.mp4"
    model_version = "v2"
    device_choice = "GPU" if torch.cuda.is_available() else "CPU"
    vad_select = True
    output_type = "Video"
    reference_speakers = ["path/to/reference_speaker1.wav", "path/to/reference_speaker2.wav"]
    models = ["model1", "model2"]
    clone = True
    original_volume = 0.5
    dubbing_volume = 1.0

    output_file, status = dub_srt(
        srt_file,
        media_file,
        model_dir,
        models,
        reference_speakers,
        model_version,
        device_choice,
        vad_select,
        output_type,
        original_volume,
        dubbing_volume,
        clone
    )
    print(f"Output file: {output_file}, Status: {status}")
