#!/usr/bin/env python3.8
import rospy
from gtts import gTTS
from pydub import AudioSegment
from next_msgs.msg import TextToSpeechAction, TextToSpeechFeedback, TextToSpeechResult
import actionlib
import os

OUTPUT_WAV = "/home/matrix/seer_pkg/src/next_text_ai/file/output_gtts.wav"
TEMP_MP3 = "/home/matrix/seer_pkg/src/next_text_ai/file/base_gtts.mp3"

def change_speed(audio, speed=1.0):
    return audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed)
    }).set_frame_rate(audio.frame_rate)

def change_pitch(audio, semitone=0):
    new_sample_rate = int(audio.frame_rate * (2.0 ** (semitone / 12.0)))
    return audio._spawn(audio.raw_data, overrides={
        "frame_rate": new_sample_rate
    }).set_frame_rate(audio.frame_rate)

class GTTSActionServer:
    def __init__(self):
        self.server = actionlib.SimpleActionServer(
            "text_to_speech",
            TextToSpeechAction,
            execute_cb=self.execute_cb,
            auto_start=False
        )
        self.server.start()
        rospy.loginfo("GTTS Action Server Ready — waiting for goal")

    def execute_cb(self, goal):
        feedback = TextToSpeechFeedback()
        result = TextToSpeechResult()

        text = goal.text.strip()
        lang = goal.lang.strip() if hasattr(goal, 'lang') and goal.lang else "th"

        if not text:
            result.success = False
            result.file_path = ""
            self.server.set_aborted(result, "Empty text")
            return

        try:
            # Feedback 1
            feedback.status = f"Generating MP3 with gTTS ({lang})"
            self.server.publish_feedback(feedback)

            # ส่งค่า lang ไปยัง gTTS
            tts = gTTS(text=text, lang=lang)
            tts.save(TEMP_MP3)

            # Feedback 2
            feedback.status = "Loading and modifying audio"
            self.server.publish_feedback(feedback)

            audio = AudioSegment.from_mp3(TEMP_MP3)
            audio = change_speed(audio, speed=goal.speed)
            audio = change_pitch(audio, semitone=goal.pitch)
            audio += goal.volume

            # Feedback 3
            feedback.status = "Exporting WAV"
            self.server.publish_feedback(feedback)

            audio.export(OUTPUT_WAV, format="wav")
            rospy.loginfo(f"Saved speech ({lang}) → {OUTPUT_WAV}")

            result.success = True
            result.file_path = OUTPUT_WAV
            self.server.set_succeeded(result)

        except Exception as e:
            result.success = False
            result.file_path = ""
            rospy.logerr(f"Error: {str(e)}")
            self.server.set_aborted(result, str(e))

def main():
    rospy.init_node("gtts_action_server")
    GTTSActionServer()
    rospy.spin()

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3.8
# import rospy
# from gtts import gTTS
# from pydub import AudioSegment
# from next_msgs.msg import TextToSpeechAction, TextToSpeechFeedback, TextToSpeechResult
# import actionlib
# import os

# OUTPUT_WAV = "/home/matrix/seer_pkg/src/next_text_ai/file/output_gtts.wav"
# TEMP_MP3 = "/home/matrix/seer_pkg/src/next_text_ai/file/base_gtts.mp3"


# def change_speed(audio, speed=1.0):
#     return audio._spawn(audio.raw_data, overrides={
#         "frame_rate": int(audio.frame_rate * speed)
#     }).set_frame_rate(audio.frame_rate)


# def change_pitch(audio, semitone=0):
#     new_sample_rate = int(audio.frame_rate * (2.0 ** (semitone / 12.0)))
#     return audio._spawn(audio.raw_data, overrides={
#         "frame_rate": new_sample_rate
#     }).set_frame_rate(audio.frame_rate)


# class GTTSActionServer:
#     def __init__(self):
#         self.server = actionlib.SimpleActionServer(
#             "text_to_speech",
#             TextToSpeechAction,
#             execute_cb=self.execute_cb,
#             auto_start=False
#         )
#         self.server.start()
#         rospy.loginfo("GTTS Action Server Ready — waiting for goal")

#     def execute_cb(self, goal):
#         feedback = TextToSpeechFeedback()
#         result = TextToSpeechResult()

#         text = goal.text.strip()

#         if not text:
#             result.success = False
#             result.file_path = ""
#             self.server.set_aborted(result, "Empty text")
#             return

#         try:
#             # Feedback 1
#             feedback.status = "Generating MP3 with gTTS"
#             self.server.publish_feedback(feedback)

#             tts = gTTS(text=text, lang="th")
#             tts.save(TEMP_MP3)

#             # Feedback 2
#             feedback.status = "Loading and modifying audio"
#             self.server.publish_feedback(feedback)

#             audio = AudioSegment.from_mp3(TEMP_MP3)

#             audio = change_speed(audio, speed=goal.speed)
#             audio = change_pitch(audio, semitone=goal.pitch)
#             audio += goal.volume

#             # Feedback 3
#             feedback.status = "Exporting WAV"
#             self.server.publish_feedback(feedback)

#             audio.export(OUTPUT_WAV, format="wav")

#             rospy.loginfo(f"Saved speech → {OUTPUT_WAV}")

#             result.success = True
#             result.file_path = OUTPUT_WAV
#             self.server.set_succeeded(result)

#         except Exception as e:
#             result.success = False
#             result.file_path = ""
#             self.server.set_aborted(result, str(e))


# def main():
#     rospy.init_node("gtts_action_server")
#     GTTSActionServer()
#     rospy.spin()


# if __name__ == "__main__":
#     main()
