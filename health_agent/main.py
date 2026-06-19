import cv2

from config import WINDOW_NAME

from camera import Camera
from face_detector import FaceDetector
from rppg_pos import POSHeartRateEstimator
from emotion_detector import EmotionDetector
from fatigue_detector import FatigueDetector
from health_agent import HealthAgent


def draw_text(frame, text, y=40):

    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


def main():

    cam = Camera()
    face_detector = FaceDetector()

    rppg = POSHeartRateEstimator()
    emotion_detector = EmotionDetector()
    fatigue_detector = FatigueDetector()
    agent = HealthAgent()

    work_time = 0
    fps_counter = 0

    while True:

        frame = cam.read()

        if frame is None:
            break

        face_data = face_detector.detect(frame)

        face_roi = None
        forehead_roi = None

        if face_data is not None:
            face_roi = face_detector.get_face_roi(frame, face_data)
            forehead_roi = face_detector.get_forehead_roi(frame, face_data)

        hr = None
        if forehead_roi is not None:
            hr = rppg.update(forehead_roi)

        emotion_result = emotion_detector.predict(face_roi)
        fatigue_result = fatigue_detector.predict(frame)

        emotion = emotion_result.get("emotion")
        fatigue = fatigue_result.get("fatigue")

        state = {
            "hr": hr,
            "emotion": emotion,
            "fatigue": fatigue,
            "work_time": work_time
        }

        advice = agent.analyze(state)

        if face_data is not None:
            frame = face_detector.draw_bbox(frame, face_data)

        if hr is not None:
            draw_text(frame, f"HR: {hr:.1f}", 40)

        if emotion is not None:
            draw_text(frame, f"Emotion: {emotion}", 70)

        if fatigue is not None:
            draw_text(frame, f"Fatigue: {fatigue:.1f}", 100)

        draw_text(frame, f"Advice: {advice}", 140)

        cv2.imshow(WINDOW_NAME, frame)

        fps_counter += 1
        if fps_counter >= 30:
            work_time += 1
            fps_counter = 0

        key = cv2.waitKey(1)
        if key == 27:
            break

    cam.release()
    face_detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()