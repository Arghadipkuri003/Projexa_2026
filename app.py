from flask import Flask, render_template, Response
import cv2

from detector import detect_objects
from emotion import detect_engagement
from attendance import mark_attendance

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame, people, phones, cleanliness = detect_objects(frame)
        engagement = detect_engagement(frame)
        present = mark_attendance(frame)

        cv2.putText(frame, f"People: {people}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(frame, f"Phones: {phones}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(frame, f"Cleanliness: {cleanliness}%", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Engagement: {engagement}%", (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template("dashboard.html")

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
