import gradio as gr
import json
import os
import subprocess
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config(config, config_path):
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def get_dataset_folders():
    dataset_root = "./dataset"
    if os.path.exists(dataset_root):
        return [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    return []

def run_finetune(language_code, project_name, output_dir, dataset_folder, model_path, 
                 num_epochs, train_batch_size):
    
    # Load base config
    config_path = "training_config_examples/finetune_mms_thai.json"
    config = load_config(config_path)
    
    # Update config with user inputs
    config["project_name"] = project_name
    config["output_dir"] = output_dir
    config["dataset_name"] = os.path.join("./dataset", dataset_folder)
    config["model_name_or_path"] = model_path
    config["num_train_epochs"] = int(num_epochs)
    config["per_device_train_batch_size"] = int(train_batch_size)
    config["per_device_eval_batch_size"] = int(train_batch_size)
    
    # Save updated config
    temp_config_path = "training_config.json"
    save_config(config, temp_config_path)
    
    # Prepare model dump if needed
    if not os.path.exists("model_dump"):
        subprocess.run([
            "python", 
            "finetune/convert_original_discriminator_checkpoint.py",
            "--language_code", language_code,
            "--pytorch_dump_folder_path", "./finetune/model_dump"
        ])
    
    # Build monotonic align if needed
    monotonic_align_dir = Path("monotonic_align/monotonic_align")
    if not monotonic_align_dir.exists():
        os.makedirs(monotonic_align_dir)
        subprocess.run([
            "python", "setup.py", "build_ext", "--inplace"
        ], cwd="monotonic_align")
    
    # Start training process
    subprocess.run([
        "accelerate", "launch", "run_vits_finetuning.py", temp_config_path
    ])
    
    return f"Training process started with configuration from {temp_config_path}!"

# Create Gradio interface
iface = gr.Interface(
    fn=run_finetune,
    inputs=[
        gr.Textbox(label="Language Code", value="tha", info="See supported language codes at: https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html"),
        gr.Textbox(label="Project Name", value="my_tts_project"),
        gr.Textbox(label="Output Path", value="./output_model"),
        gr.Dropdown(label="Dataset Folder", choices=get_dataset_folders(), value=get_dataset_folders()[0] if get_dataset_folders() else None),
        gr.Textbox(label="Pretraied Model Path", value="./model_dump"),
        gr.Number(label="Number of Training Epochs", value=200),
        gr.Number(label="Batch Size", value=8)
    ],
    outputs="text",
    submit_btn="Start Finetuning",
    title="Finetune MMS-TTS-VITS Interface",
    description="Configure and prepare MMS-TTS model finetuning"
)

if __name__ == "__main__":
    iface.launch()
