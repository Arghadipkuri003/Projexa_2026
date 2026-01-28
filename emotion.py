from fer import FER

emotion_detector = FER()

def detect_engagement(frame):
    emotions = emotion_detector.detect_emotions(frame)
    engaged = 0

    for e in emotions:
        dominant = max(e['emotions'], key=e['emotions'].get)
        if dominant in ["happy", "neutral"]:
            engaged += 1

    total = len(emotions)
    engagement = int((engaged / total) * 100) if total > 0 else 0
    return engagement
