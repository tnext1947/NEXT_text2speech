#!/usr/bin/env python3.8
import rospy
from std_msgs.msg import String
import socket

HOST = "127.0.0.1"
PORT = 6000

def callback(msg):
    text = msg.data.strip()
    if not text:
        return

    rospy.loginfo(f"Forwarding TTS text -> {text}")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        s.sendall(text.encode("utf-8"))
        s.close()
    except Exception as e:
        rospy.logerr(f"Forwarding error: {e}")

def main():
    rospy.init_node("tts_forwarder")
    rospy.Subscriber("/speak", String, callback)
    rospy.loginfo("Forwarder ready — Listening /speak")
    rospy.spin()

if __name__ == "__main__":
    main()
