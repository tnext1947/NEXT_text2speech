import os
import torch
from transformers import VitsModel, VitsTokenizer
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import scipy
from pathlib import Path
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

models = {}
tokenizers = {}

def get_model_names(model_dir):
    model_paths = Path(model_dir).glob('*')
    return [model_path.name for model_path in model_paths if model_path.is_dir()]

def load_vits_model(model_name, model_dir):
    if model_name not in models:
        model_path = os.path.join(model_dir, model_name)
        models[model_name] = VitsModel.from_pretrained(model_path).to(device)
        tokenizers[model_name] = VitsTokenizer.from_pretrained(model_path)
    return models[model_name], tokenizers[model_name]

def generate_speech(text, model_dir, model_name, speaking_rate=1.0):
    model, tokenizer = load_vits_model(model_name, model_dir)
    processed_string = (text)
    inputs = tokenizer(processed_string, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    #set_seed(456)
    
    # Set model parameters
    model.speaking_rate = speaking_rate
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Move output back to CPU for audio processing
    waveform = outputs.waveform[0].cpu().numpy()
    
    # Ensure correct sampling rate
    if hasattr(model.config, 'sampling_rate'):
        sampling_rate = model.config.sampling_rate
    else:
        sampling_rate = 48000
    
    return sampling_rate, waveform

def save_audio(sampling_rate, audio_data, filename="output.wav"):
    scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_data)
    return filename

def voice_cloning(base_speaker, reference_speaker, model_version, device_choice, vad_select):
    try:
        ckpt_converter = f'./OPENVOICE_MODELS/{model_version}'
        device = "cuda:0" if device_choice == "GPU" and torch.cuda.is_available() else "cpu"
        
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        source_se, _ = se_extractor.get_se(base_speaker, tone_color_converter, vad=vad_select)
        target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, vad=vad_select)
        
        save_path = f'{output_dir}/output_cloned_podtts.wav'
        
        tone_color_converter.convert(
            audio_src_path=base_speaker, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
        )
        return save_path, "Voice cloning successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"

def generate_podcast_script(script, model_dir, model_names, speaking_rate=1.0, noise_scale=1, clone=False, reference_speakers=None, model_version="v2", device_choice="GPU", vad_select=False):
    lines = script.split('\n')
    audio_segments = []
    
    for line in lines:
        if line.strip():
            speaker, text = line.split(':', 1)
            model_name = model_names[int(speaker.strip().split()[-1]) - 1]
            sr, audio = generate_speech(text.strip(), model_dir, model_name, speaking_rate)
            audio_segments.append(audio)
    
    combined_audio = np.concatenate(audio_segments)
    output_path = f'{output_dir}/podcast_output.wav'
    save_audio(sr, combined_audio, output_path)
    
    if clone and reference_speakers:
        cloned_audio_file, status = voice_cloning(output_path, reference_speakers[0], model_version, device_choice, vad_select)
        return cloned_audio_file, status
    
    return output_path, "Podcast generation successful!"
