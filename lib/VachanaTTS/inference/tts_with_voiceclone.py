import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from transformers import VitsModel, VitsTokenizer
import scipy
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

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
    model.sentence_silence = 0
    #model.energy = 10
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Move output back to CPU for audio processing
    waveform = outputs.waveform[0].cpu().numpy()
    
    #resampled_audio = resample_poly(waveform, 16000, 16000)  # Assuming the original sampling rate is 22050
    sampling_rate = model.config.sampling_rate

    # Ensure correct sampling rate
    #if hasattr(model.config, 'sampling_rate'):
    #    sampling_rate = model.config.sampling_rate
    #else:
    #    sampling_rate = 48000
    
    return sampling_rate, waveform

def save_audio(sampling_rate, audio_data, filename="output_tts.wav"):
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
        
        save_path = f'{output_dir}/output_cloned_tts.wav'
        
        tone_color_converter.convert(
            audio_src_path=base_speaker, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
        )
        return save_path,"Voice cloning successful!"
    
    except Exception as e:
        return None, f"Error: {str(e)}"
