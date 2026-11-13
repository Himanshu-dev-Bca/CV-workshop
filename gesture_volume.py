
import math
import sys

import cv2
import mediapipe as mp
import numpy as np
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities
from pycaw.pycaw import IAudioEndpointVolume


def initialize_volume_control():
    if sys.platform != "win32":
        raise RuntimeError("This script requires Windows to control the system volume.")

    try:
        devices = AudioUtilities.GetSpeakers()
        if hasattr(devices, "EndpointVolume"):
            volume = devices.EndpointVolume
        else:
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None
            )
            volume = cast(interface, POINTER(IAudioEndpointVolume))
        min_vol, max_vol, _ = volume.GetVolumeRange()
        return volume, min_vol, max_vol
    except Exception as exc:
        raise RuntimeError(f"Unable to access the Windows audio device: {exc}") from exc


def main():
    try:
        volume, min_vol, max_vol = initialize_volume_control()
    except RuntimeError as exc:
        print(exc)
        return 1

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    draw = mp.solutions.drawing_utils
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Unable to open the webcam. Please connect a camera and try again.")
        hands.close()
        return 1

    previous_volume = -1

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read frames from the webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                    h, w, _ = frame.shape

                    thumb = hand.landmark[4]
                    index = hand.landmark[8]

                    x1, y1 = int(thumb.x * w), int(thumb.y * h)
                    x2, y2 = int(index.x * w), int(index.y * h)

                    distance = math.hypot(x2 - x1, y2 - y1)

                    target = np.interp(distance, [30, 200], [min_vol, max_vol])
                    target = float(np.clip(target, min_vol, max_vol))

                    if previous_volume == -1:
                        previous_volume = target

                    current = previous_volume * 0.7 + target * 0.3

                    volume.SetMasterVolumeLevel(current, None)
                    previous_volume = current

                    percentage = int(np.interp(current, [min_vol, max_vol], [0, 100]))

                    cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    cv2.putText(
                        frame,
                        f"Volume: {percentage}%",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    bar = int(np.interp(percentage, [0, 100], [400, 150]))
                    cv2.rectangle(frame, (30, 150), (60, 400), (255, 255, 255), 2)
                    cv2.rectangle(frame, (30, bar), (60, 400), (0, 255, 0), cv2.FILLED)

            cv2.imshow("AI Hand Gesture Volume Controller", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        hands.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
