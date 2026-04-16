import os
import re
from pythainlp import word_tokenize

# Ensure UTF-8 encoding is set
os.environ['PYTHONIOENCODING'] = 'utf-8'

def english_to_thai_fallback(word):
    mapping = {
        "today": "ทูเด",
        "hello": "เฮลโล",
        "world": "เวิลด์",
        "computer": "คอมพิวเตอร์",
        "phone": "โฟน",
        "school": "สคูล",
        "teacher": "ทีเชอร์",
        "student": "สตูเดนท์",
        "apple": "แอปเปิล",
        "orange": "ออเรนจ์",
        "table": "เทเบิล",
        "chair": "แชร์",
        "window": "วินโดว์",
        "door": "ดอร์",
        "water": "วอเทอร์",
        "coffee": "คอฟฟี่",
        "milk": "มิลค์",
        "juice": "จูซ",
        "food": "ฟูด",
        "car": "คาร์",
        "bus": "บัส",
        "train": "เทรน",
        "airplane": "แอร์เพลน",
        "boat": "โบ๊ท",
        "dog": "ด็อก",
        "cat": "แคท",
        "bird": "เบิร์ด",
        "fish": "ฟิช",
        "house": "เฮ้าส์",
        "city": "ซิตี้",
        "country": "คันทรี",
        "family": "แฟมิลี",
        "friend": "เฟรนด์",
        "love": "เลิฟ",
        "happiness": "แฮปปิเนส",
        "sadness": "แซดเนส",
        "anger": "แองเกอร์",
        "smile": "สไมล์",
        "cry": "คราย",
        "laugh": "ลาฟ",
        "light": "ไลท์",
        "dark": "ดาร์ก",
        "sun": "ซัน",
        "moon": "มูน",
        "star": "สตาร์",
        "ocean": "โอเชียน",
        "mountain": "เมาเทน",
        "river": "ริเวอร์",
        "forest": "ฟอเรสต์",
        "i": "ไอ",
        "love": "เลิฟ",
        "you": "ยู",
        "talk": "ทอล์ก",
        "sing": "ซิง",
        "dance": "แดนซ์",
        "read": "รีด",
        "write": "ไรท์",
        "run": "รัน",
        "walk": "วอล์ค",
        "jump": "จัมป์",
        "swim": "สวิม",
        "eat": "อีท",
        "drink": "ดริงค์",
        "sleep": "สลีป",
        "wake": "เวค",
        "good": "กู๊ด",
        "bad": "แบด",
        "happy": "แฮปปี้",
        "sad": "แซด",
        "angry": "แองกรี",
        "tired": "ไทร์ด"
    }
    return mapping.get(word.lower(), word)

def clean_thai_text(text):
    def replace_mai_ek(match):
        return match.group(1) + '\u0E4D' + 'า'  # Replace ำ with ํ + า

    # Replace occurrences of ำ with ํา
    text = re.sub(r'([ก-ฮ])ำ', replace_mai_ek, text)
    
    # Tokenize the text
    words = word_tokenize(text, keep_whitespace=True)

    # Convert English words to Thai phonemes
    cleaned_text = []
    for word in words:
        if re.search(r'[a-zA-Z]', word):  # If the word contains English letters
            try:
                from pythainlp import transliterate  # Import here to handle the library conditionally
                thai_phoneme = transliterate(word, engine='ipa')
                cleaned_text.append(thai_phoneme)
            except Exception:
                cleaned_text.append(english_to_thai_fallback(word))
        else:
            cleaned_text.append(word)

    return ''.join(cleaned_text)
