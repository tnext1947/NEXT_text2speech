#!/usr/bin/env python3.8
import socket
from vachanatts import TTS
from pydub import AudioSegment

HOST = "127.0.0.1"
PORT = 6000

OUTPUT_FILE = "/home/ter/ai/src/next_text_ai/file/output_vacha.wav"
BASE_FILE = "/home/ter/ai/src/next_text_ai/file/base_vacha.wav"
## rostopic pub /speak std_msgs/String "data: 'หุ่นยนต์มาถึงแล้ว กรุณากดปุ่ม Confirm ด้วยค่ะ|voice:th_m_1'" ##

def parse_request(data):
    """
    Parse messages like:
    'ข้อความ|voice:th_m_1'
    """
    text = data
    voice = "th_f_1"   # default voice

    if "|" in data:
        parts = data.split("|")
        text = parts[0]

        for p in parts[1:]:
            if p.startswith("voice:"):
                voice = p.replace("voice:", "").strip()

    return text.strip(), voice.strip()


def speak(text, voice="th_f_1", speed=0.9, volume=1.0, pitch=0.85):

    # 1. Generate raw Vachana TTS
    TTS(
        text=text,
        voice=voice,
        output=BASE_FILE,
        volume=volume,
        speed=speed
    )

    # 2. Post-process pitch
    audio = AudioSegment.from_wav(BASE_FILE)

    if pitch != 1.0:
        audio = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": int(audio.frame_rate * pitch)}
        ).set_frame_rate(audio.frame_rate)

    audio.export(OUTPUT_FILE, format="wav")
    print(f"[Worker] Generated ({voice}) → {OUTPUT_FILE}")


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)

    print(f"[Worker] Listening on port {PORT} ...")

    while True:
        conn, addr = s.accept()
        data = conn.recv(4096).decode("utf-8")
        conn.close()

        if data:
            print(f"[Worker] Input: {data}")
            text, voice = parse_request(data)
            print(f"[Worker] Parsed text='{text}', voice='{voice}'")

            speak(text, voice=voice)


if __name__ == "__main__":
    main()
