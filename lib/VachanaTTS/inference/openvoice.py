import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# Output directory setup
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

# Function for voice cloning
def voice_cloning(base_speaker, reference_speaker, model_version, device_choice, vad_select):
    try:
        # Determine paths and device
        ckpt_converter = f'./OPENVOICE_MODELS/{model_version}'
        device = "cuda:0" if device_choice == "GPU" and torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        # Load the ToneColorConverter
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

        # Extract speaker embeddings
        source_se, _ = se_extractor.get_se(base_speaker, tone_color_converter, vad=vad_select)
        target_se, _ = se_extractor.get_se(reference_speaker, tone_color_converter, vad=vad_select)
        
        # Define output file paths
        save_path = f'{output_dir}/output_cloned.wav'
        
        # Perform tone color conversion
        tone_color_converter.convert(
            audio_src_path=base_speaker, 
            src_se=source_se, 
            tgt_se=target_se, 
            output_path=save_path,
        )
        return save_path, "Voice cloning successful!"
    except Exception as e:
        return None, f"Error: {str(e)}"
