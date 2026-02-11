import cv2
import mediapipe as mp
import math


class HandDetector():

    def __init__(self, mode=False, maxHands=1,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )

        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]  # Thumb + 4 fingers
        self.results = None
        self.lmList = []
        self.bbox = []
        self.currentHandType = None

    # -------------------------------------------------
    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms,
                        self.mpHands.HAND_CONNECTIONS)

        return img

    # -------------------------------------------------
    def findPosition(self, img, handNo=0, draw=True):

        self.lmList = []
        self.bbox = []

        if self.results and self.results.multi_hand_landmarks:

            if handNo >= len(self.results.multi_hand_landmarks):
                return self.lmList, self.bbox

            myHand = self.results.multi_hand_landmarks[handNo]

            xList = []
            yList = []

            h, w, _ = img.shape

            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy),
                               4, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            self.bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img,
                              (xmin - 20, ymin - 20),
                              (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

            # Detect Hand Type
            if self.results.multi_handedness:
                self.currentHandType = \
                    self.results.multi_handedness[handNo] \
                    .classification[0].label

        return self.lmList, self.bbox

    # -------------------------------------------------
    def fingersUp(self):

        fingers = []

        if len(self.lmList) == 0:
            return fingers

        # Thumb (special case)
        if self.currentHandType == "Right":
            fingers.append(
                1 if self.lmList[self.tipIds[0]][1] >
                self.lmList[self.tipIds[0] - 1][1] else 0)
        else:
            fingers.append(
                1 if self.lmList[self.tipIds[0]][1] <
                self.lmList[self.tipIds[0] - 1][1] else 0)

        # Other 4 fingers
        for id in range(1, 5):
            fingers.append(
                1 if self.lmList[self.tipIds[id]][2] <
                self.lmList[self.tipIds[id] - 2][2] else 0)

        return fingers

    # -------------------------------------------------
    def findDistance(self, p1, p2, img=None, draw=True):

        if len(self.lmList) == 0:
            return 0, img, []

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)

        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2),
                     (255, 0, 255), 2)
            cv2.circle(img, (x1, y1),
                       6, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2),
                       6, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy),
                       6, (0, 255, 0), cv2.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]

    # -------------------------------------------------
    def getHandType(self):
        return self.currentHandType
