import cv2
import numpy as np
import time
import math
import HandTrackingModule as htm
import screen_brightness_control as sbc

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.HandDetector(detectionCon=0.7, trackCon=0.7)

# ---------------- Variables ----------------
ptime = 0
brightnessLocked = False
lastTapTime = 0
tapCount = 0
DOUBLE_TAP_DELAY = 0.5

currentBrightness = sbc.get_brightness()[0]

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        # Thumb & Index for brightness control
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Index & Middle for double tap detection
        x3, y3 = lmList[12][1], lmList[12][2]

        # Draw
        cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)

        # -------- Brightness Control --------
        if not brightnessLocked:
            length = math.hypot(x2 - x1, y2 - y1)
            brightness = np.interp(length, [50, 250], [10, 90])
            brightness = int(brightness)
            brightness = max(10, min(90, brightness))

            sbc.set_brightness(brightness)
            currentBrightness = brightness

        # -------- Double Tap Detection --------
        tapLength = math.hypot(x2 - x3, y2 - y3)

        if tapLength < 30:   # fingers touching
            currentTime = time.time()

            if currentTime - lastTapTime < DOUBLE_TAP_DELAY:
                tapCount += 1
            else:
                tapCount = 1

            lastTapTime = currentTime

            if tapCount == 2:
                brightnessLocked = not brightnessLocked
                tapCount = 0
                time.sleep(0.3)   # prevent multiple toggles

        # -------- UI --------
        cv2.putText(img,
                    "LOCKED" if brightnessLocked else "UNLOCKED",
                    (400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255) if brightnessLocked else (0, 255, 0),
                    3)

        cv2.putText(img,
                    f'{currentBrightness} %',
                    (50, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    3)

    # FPS
    ctime = time.time()
    fps = 1 / (ctime - ptime) if (ctime - ptime) != 0 else 0
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Brightness Control with Lock", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
