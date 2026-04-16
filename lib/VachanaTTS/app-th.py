import gradio as gr
import argparse
from inference.tts_with_voiceclone import generate_speech, save_audio, get_model_names, voice_cloning
from inference.openvoice import voice_cloning as vc_voice_cloning
from inference.dubbing import dub_srt, get_model_names as get_dubbing_model_names
from inference.podcast import generate_podcast_script, get_model_names as get_podcast_model_names
from inference.thaicleantext import clean_thai_text
import os

model_dir = "./models"
model_names = get_model_names(model_dir)

def create_tts_interface():
    
    with gr.Column():
        gr.Markdown("## เครื่องมือแปลงข้อความเป็นเสียง พร้อม การโคลนเสียง")
        gr.Markdown("โมเดล finetune จาก MMS-TTS และการโคลนเสียงด้วย OpenVoice ")
        
        with gr.Row():
            text_input = gr.Textbox(label="แปลงข้อความเป็นเสียง", lines=3)
        
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(model_names, label="โมเดล", interactive=True)
                speaking_rate = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="ความเร็วในการพูด")

            with gr.Column():
                model_name_input = gr.Textbox(label="ดาวน์โหลด โมเดล", placeholder="e.g., VIZINTZOR/MMS-TTS-THAI-MALEV1")
                download_btn = gr.Button("ดาวน์โหลด")

            with gr.Column():
                clone_checkbox = gr.Checkbox(label="โคลนเสียง", value=False)
                ref_audio = gr.Audio(label="เสียง ตัวอย่าง", type="filepath")
                model_version = gr.Dropdown(["v1", "v2"], value="v2", label="Model Version")
                device = gr.Dropdown(["CPU", "GPU"], value="GPU", label="Device")
                vad = gr.Checkbox(label="VAD", value=True,info="สามารถใช้ได้กับเสียงและข้อความที่มีความยาว 5 วินาทีขึ้นไป")
        
        generate_btn = gr.Button("สร้าง")
        output_audio = gr.Audio(label="เสียง")
        status = gr.Textbox(label="สถานะ")
        
        def download_model(model_name):
            os.system(f"git clone https://huggingface.co/{model_name} {model_dir}/{model_name.split('/')[-1]}")
            return f"Model {model_name} ดาวน์โหลด โมเดลเสียงแล้ว!", gr.update(choices=get_model_names(model_dir))
        
        def ui_fn(text, model_name, speaking_rate, clone, reference_speaker, model_version, device_choice, vad_select):
            cleaned_text = clean_thai_text(text)
            sampling_rate, audio_data = generate_speech(cleaned_text, model_dir, model_name, speaking_rate)
            audio_file = save_audio(sampling_rate, audio_data)
            
            if clone:
                cloned_audio_file, status = voice_cloning(audio_file, reference_speaker, model_version, device_choice, vad_select)
                return cloned_audio_file, status
            else:
                return audio_file, "สร้างคำพูดสำเร็จ!"
        
        download_btn.click(
            download_model,
            inputs=[model_name_input],
            outputs=[status, model]
        )
        
        generate_btn.click(
            ui_fn,
            inputs=[text_input, model, speaking_rate, clone_checkbox, ref_audio, model_version, device, vad],
            outputs=[output_audio, status]
        )

def create_vc_interface():
    with gr.Column():
        gr.Markdown("## การโคลนเสียง")
        gr.Markdown("การโคลนเสียงจากเสียงด้วย OpenVoice")

        with gr.Row():
            base_speaker = gr.Audio(label="เสียงหลัก", type="filepath")
            reference_speaker = gr.Audio(label="เสียงเป้าหมาย", type="filepath")
        
        with gr.Row():
            model_version = gr.Dropdown(["v1", "v2"], value="v2", label="Model Version")
            device = gr.Dropdown(["CPU", "GPU"], value="GPU", label="Device")
            vad = gr.Checkbox(label="VAD", value=False,info="สามารถใช้ได้กับเสียงและข้อความที่มีความยาว 5 วินาทีขึ้นไป")
        
        clone_btn = gr.Button("โคลน")
        output_audio = gr.Audio(label="เสียง โคลน")
        status = gr.Textbox(label="สถานะ")
        
        def handle_clone(base_speaker, reference_speaker, model_version, device_choice, vad_select):
            result, status = vc_voice_cloning(base_speaker, reference_speaker, model_version, device_choice, vad_select)
            return result, status
        
        clone_btn.click(
            handle_clone,
            inputs=[base_speaker, reference_speaker, model_version, device, vad],
            outputs=[output_audio, status]
        )

def create_dubbing_interface():
    
    with gr.Column():
        gr.Markdown("## พากย์เสียง")
        gr.Markdown("สร้างเสียงหรือวิดีโอที่พากย์จากไฟล์ SRT และวิดีโอหรือเสียงต้นฉบับ ด้วยการโคลนเสียง")

        with gr.Row():
            srt_file = gr.File(label="SRT ไฟล์")
            media_file = gr.File(label="วิดีโอ/เสียง", type="filepath")
        
        with gr.Row():
            num_speakers = gr.Slider(1, 4, 1, step=1, label="จำนวนผู้พูด")
        
        speaker_rows = []
        reference_speakers = []
        model_inputs = []
        
        for i in range(4):
            with gr.Row(visible=(i == 0)) as row:
                model = gr.Dropdown(choices=model_names, label=f"โมเดล {i+1}" if i > 0 else "โมเดล")
                ref_audio = gr.Audio(label=f"เสียงโคลน {i+1}" if i > 0 else "เสียงโคลน", type="filepath")
                
                model_inputs.append(model)
                reference_speakers.append(ref_audio)
                speaker_rows.append(row)
        
        with gr.Row():
            model_version = gr.Dropdown(["v1", "v2"], value="v2", label="Model Version")
            device = gr.Dropdown(["CPU", "GPU"], value="GPU", label="Device")
            vad = gr.Checkbox(label="VAD", value=True,info="สามารถใช้ได้กับเสียงและข้อความที่มีความยาว 5 วินาทีขึ้นไป")
            output_type = gr.Dropdown(["Audio", "Video"], value="Video", label="ประเภท ส่งออก")
            clone_checkbox = gr.Checkbox(label="โคลนเสียง", value=False)
            original_value = gr.Slider(0.0, 1.0, 0.5, step=0.1, value=0.5,label="ความดังเสียงต้นฉบับ", interactive=True)
            dubbing_value = gr.Slider(0.0, 1.0, 1.0, step=0.1, value=2,label="ความดังเสียงพากย์", interactive=True)
            refresh_btn = gr.Button("รีเฟรช", size="sm")

        def refresh():
            updated_choices = get_model_names(model_dir)
            return [gr.update(choices=updated_choices) for _ in range(4)]
        
        num_speakers.change(
            lambda x: [gr.Row.update(visible=i < x) for i in range(4)],
            [num_speakers],
            speaker_rows
        )
        
        dub_btn = gr.Button("สร้างการพากย์")
        output_file = gr.File(label="ผลลัพธ์")
        status = gr.Textbox(label="สถานะ")
        
        def ui_fn(srt_file, media_file, num_speakers, model_version, device_choice, vad_select, output_type, clone, original_value, dubbing_value, *args):
            mid = len(args) // 2
            reference_speakers = args[:mid]
            models = args[mid:]
            
            reference_speakers = [ref for ref in reference_speakers[:num_speakers] if ref is not None]
            models = [model for model in models[:num_speakers] if model is not None]
            
            if not models:
                return None, "ไม่พบ โมเดลที่เลือก"
            
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
                original_value,  # Original volume
                dubbing_value,  # Dubbing volume
                clone
            )
            return output_file, status
        
        refresh_btn.click(
            refresh,
            outputs=model_inputs
        )

        dub_btn.click(
            ui_fn,
            inputs=[srt_file, media_file, num_speakers, model_version, device, vad, output_type, clone_checkbox, original_value, dubbing_value, *reference_speakers, *model_inputs],
            outputs=[output_file, status]
        )

def create_podcast_interface():
    
    with gr.Column():
        gr.Markdown("## พอดแคสต์์")
        gr.Markdown("สร้างเสียงพอดแคสต์จากสคริปต์ ตัวอย่าง: Speaker 1: สวัสดีครับ\nSpeaker 2: สวัสดีค่ะ\n...")

        with gr.Row():
            script_input = gr.Textbox(label="บทความ", lines=10, placeholder="Speaker 1: สวัสดีครับ\nSpeaker 2: สวัสดีค่ะ\n...")
        
        with gr.Row():
            with gr.Column():
                model1 = gr.Dropdown(model_names, label="โมเดล 1")
                model2 = gr.Dropdown(model_names, label="โมเดล 2")
                speaking_rate = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="ความเร็วในการพูด")
                refresh_btn = gr.Button("รีเฟรช", size="md")
                
        def refresh():
            updated_choices = get_model_names(model_dir)
            return [gr.update(choices=updated_choices) for _ in range(2)]
        
        generate_btn = gr.Button("สร้างพอดแคสต์")
        output_audio = gr.Audio(label="เสียงพอดแคสต์")
        status = gr.Textbox(label="สถานะ")
        
        def ui_fn(script, model1, model2, speaking_rate):
            cleaned_script = clean_thai_text(script)
            model_names = [model1, model2]
            output_file, status = generate_podcast_script(cleaned_script, model_dir, model_names, speaking_rate)
            return output_file, status
        
        refresh_btn.click(
            refresh,
            outputs=[model1, model2]
        )

        generate_btn.click(
            ui_fn,
            inputs=[script_input, model1, model2, speaking_rate],
            outputs=[output_audio, status]
        )

def create_app():
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# VachanaTTS")
        
        with gr.Tabs():
            with gr.Tab("แปลงข้อความเป็นเสียง"):
                create_tts_interface()
            with gr.Tab("การโคลนเสียง"):
                create_vc_interface()
            with gr.Tab("พอดแคสต์"):
                create_podcast_interface()
            with gr.Tab("การพากย์เสียง"):
                create_dubbing_interface()
    
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", type=bool, default=False, help="Enable Gradio share mode")
    args = parser.parse_args()
    
    app = create_app()
    app.queue()
    app.launch(inbrowser=True, share=args.share)
