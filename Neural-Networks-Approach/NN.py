import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pyautogui

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    result = recognizer.recognize(mp_image)
    if result.gestures:
        gesture = result.gestures[0][0].category_name
        cv2.putText(img, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2, cv2.LINE_AA)

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if gesture == 'Closed_Fist':
            x = hand_landmarks.landmark[12].x * pyautogui.size().width
            y = hand_landmarks.landmark[12].y * pyautogui.size().height
            pyautogui.moveTo(x, y)
        elif gesture == 'Thumb_Up':
            pyautogui.hotkey('ctrl', '+')
        elif gesture == 'Thumb_Down':
            pyautogui.hotkey('ctrl', '-')
        elif gesture == 'Victory':
            pyautogui.leftClick()
        elif gesture == 'ILoveYou':
            pyautogui.rightClick()

    cv2.imshow("Hand Tracker", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
