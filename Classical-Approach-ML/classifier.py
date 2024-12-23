import os
import numpy as np
import cv2
import pickle
import util
import pyautogui
import customtkinter as ctk
from PIL import Image

class HandGestureRecognizer:

    def __init__(self, model_path=os.path.join(util.script_dir, util.MODEL_NAME)) -> None:
        '''Initialize the HandGestureRecognizer with the trained model'''
        # Read the model and save it in a variable
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)['model']

    def recognize_gesture(self, binary_image: np.ndarray) -> str:
        '''
        Recognize the gesture from the binary image by extracting HOG features then predicting the gesture
        '''
        # Extract HOG features
        features = util.extract_hog_features(binary_image)
        if features is None:
            return "No hand detected"

        # Predict the gesture
        prediction = self.classifier.predict([np.asarray(features)])
        return prediction[0]

    def take_action(self, gesture: str, center: tuple, canvas: ctk.CTkCanvas) -> None:
        '''Take action based on the recognized gesture within the GUI'''
        if gesture == 'closed_fist':
            # Move a pointer on the canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            pos_x = center[0] * canvas_width / (util.x2 - util.x1)
            pos_y = center[1] * canvas_height / (util.y2 - util.y1)
            canvas.delete("pointer")
            canvas.create_oval(pos_x - 5, pos_y - 5, pos_x + 5, pos_y + 5, fill="red", tags="pointer")

class HandGestureApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition GUI")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2a2d2e")

        # Header Label
        self.header = ctk.CTkLabel(self.root, text="Hand Gesture Recognition", font=ctk.CTkFont(size=24, weight="bold"))
        self.header.grid(row=0, column=0, columnspan=3, pady=20, sticky="n")
        
        # Set up the recognizer
        self.recognizer = HandGestureRecognizer()

        # Set up the camera
        self.cap = cv2.VideoCapture(0)

        # Frame for the video feed
        self.video_frame = ctk.CTkLabel(self.root, text="")
        self.video_frame.grid(row=1, column=0, padx=20, pady=20)

        # Frame for the ROI feed
        self.roi_frame = ctk.CTkLabel(self.root, text="")
        self.roi_frame.grid(row=1, column=1, padx=20, pady=(40, 0))

        # Gesture Label directly above ROI
        self.gesture_label = ctk.CTkLabel(self.root, text="Gesture: None", font=ctk.CTkFont(size=20, weight="bold"))
        self.gesture_label.grid(row=1, column=1, sticky="n", pady=(10, 0))

        # Canvas for gesture actions
        self.canvas_frame = ctk.CTkFrame(self.root)
        self.canvas_frame.grid(row=1, column=2, padx=20, pady=20)
        self.canvas_label = ctk.CTkLabel(self.canvas_frame, text="Gesture Actions", font=ctk.CTkFont(size=18))
        self.canvas_label.pack(pady=10)
        self.canvas = ctk.CTkCanvas(self.canvas_frame, width=400, height=400, bg="white")
        self.canvas.pack(pady=10)

        # Footer with instructions
        self.footer = ctk.CTkLabel(self.root, text="\u2022 Perform gestures within the ROI box \u2022 Use 'q' to quit the application \u2022 Enjoy interacting with gestures!", \
                                  font=ctk.CTkFont(size=16), anchor="w", wraplength=1000, justify="left")
        self.footer.grid(row=2, column=0, columnspan=3, pady=20, sticky="w")

        # Update loop
        self.update()

    def update(self):
        '''Update the video feed and process gestures'''
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)

        # Draw the ROI
        cv2.rectangle(frame, (util.x1 - 1, util.y1 - 1), (util.x2 + 1, util.y2 + 1), (0, 255, 0), 2)
        roi = frame[util.x1:util.x2, util.y1:util.y2]

        # Segment the image using k-means
        roi_kmeans, center = util.segment_hand_kmeans(roi, 3)

        # Recognize the gesture
        gesture = self.recognizer.recognize_gesture(roi_kmeans)

        # Update gesture label
        self.gesture_label.configure(text=f"Gesture: {gesture}")

        # Perform action on canvas
        self.recognizer.take_action(gesture, center, self.canvas)

        # Convert the frame and ROI for customtkinter display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_rgb = cv2.cvtColor(roi_kmeans, cv2.COLOR_BGR2RGB)
        frame_img_pil = Image.fromarray(frame_rgb)
        roi_img_pil = Image.fromarray(roi_rgb)
        frame_img_ctk = ctk.CTkImage(frame_img_pil, size=(640, 480))
        roi_img_ctk = ctk.CTkImage(roi_img_pil, size=(320, 240))

        self.video_frame.configure(image=frame_img_ctk)
        self.video_frame.image = frame_img_ctk

        self.roi_frame.configure(image=roi_img_ctk)
        self.roi_frame.image = roi_img_ctk

        # Schedule the next update
        self.root.after(10, self.update)

    def on_closing(self):
        '''Release resources on closing'''
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == '__main__':
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    app = HandGestureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
