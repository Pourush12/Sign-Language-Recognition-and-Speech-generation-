import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from gtts import gTTS
import pygame
import tempfile
import os
import time


class GestureRecognitionApp:
    def __init__(self, root, model_path='./model.p'):
        self.root = root
        self.root.title("Gesture Recognition")

        self.model_dict = pickle.load(open(model_path, 'rb'))
        self.model = self.model_dict['model']

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        self.labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
                            11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
                            21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

        self.cap = cv2.VideoCapture(0)

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.predicted_label = tk.Label(self.root, text="Predicted Sign: None", font=("Helvetica", 16))
        self.predicted_label.pack()

        self.predicted_string = ""
        self.predicted_string_label = tk.Label(self.root, text="Predicted String: ", font=("Helvetica", 16))
        self.predicted_string_label.pack()

        self.language_label = tk.Label(self.root, text="Select Language:", font=("Helvetica", 12))
        self.language_label.pack()

        self.languages = ['en', 'fr', 'es', 'de']  # Language codes
        self.language_names = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German'
        }  # Dictionary mapping language codes to their full names

        # Create a list of full language names for display in the dropdown menu
        language_options = [self.language_names[code] for code in self.languages]

        self.language_var = tk.StringVar()
        self.language_dropdown = ttk.Combobox(self.root, textvariable=self.language_var, values=language_options,
                                              state="readonly")
        self.language_dropdown.pack()
        self.language_dropdown.set('English')  # Default language is English

        self.convert_button = tk.Button(self.root, text="Convert to Speech", command=self.convert_to_speech)
        self.convert_button.pack()

        self.last_prediction_time = None  # Keep track of the last prediction time
        self.paused = False  # Flag to indicate if prediction is paused
        self.last_predicted_sign = ""  # Keep track of the last predicted sign before pausing

        self.root.bind("<space>", self.toggle_pause)  # Bind space key to toggle pause
        self.root.bind("r", self.reset_string)  # Bind 'r' key to reset string
        self.root.bind("<BackSpace>", self.remove_last_character)

        pygame.init()  # Initialize pygame for audio playback

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    self.mp_hands.HAND_CONNECTIONS,  # hand connections
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = self.model.predict([np.asarray(data_aux)])
            predicted_character = self.labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

            self.predicted_label.config(text=f"Predicted Sign: {predicted_character}")

            current_time = time.time()
            if not self.paused and (self.last_prediction_time is None or current_time - self.last_prediction_time >= 5):
                # Capture the predicted sign if it's not paused and it's the first prediction or 5 seconds have passed since the last capture
                self.predicted_string += predicted_character
                self.predicted_string_label.config(text=f"Predicted String: {self.predicted_string}")
                self.last_prediction_time = current_time

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.video_label.after(10, self.update_frame)

    def toggle_pause(self, event):
        # Toggle pause state
        self.paused = not self.paused
        if not self.paused:
            # Resume prediction
            self.last_prediction_time = None  # Reset last prediction time
            if self.last_predicted_sign:
                # Append the last predicted sign before pausing
                self.predicted_string += self.last_predicted_sign
                self.predicted_string_label.config(text=f"Predicted String: {self.predicted_string}")
                self.last_predicted_sign = ""  # Reset the last predicted sign


    def reset_string(self, event):
        # Reset the predicted string
        self.predicted_string = ""
        self.predicted_string_label.config(text="Predicted String: ")
        # Stop the audio playback
        pygame.mixer.music.stop()

    def remove_last_character(self, event):
        # Remove the last character from the predicted string
        self.predicted_string = self.predicted_string[:-1]
        self.update_predicted_string_label()

    def update_predicted_string_label(self):
        # Update the label to display the updated predicted string
        self.predicted_string_label.config(text=f"Predicted String: {self.predicted_string}")

    def convert_to_speech(self):
        # Convert the predicted string to speech in the selected language
        selected_language = self.language_var.get()  # Get the full name of the selected language
        language_code = [code for code, name in self.language_names.items() if name == selected_language][0]
        tts = gTTS(text=self.predicted_string, lang=language_code)

        # Save the speech as a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_file.close()
        tts.save(temp_file.name)

        # Load the temporary file into Pygame mixer and play it
        sound = pygame.mixer.Sound(temp_file.name)
        sound.play()

        # Remove the temporary file after playback
        os.remove(temp_file.name)


if __name__ == "__main__":
    import time  # Import time module

    root = tk.Tk()
    app = GestureRecognitionApp(root)
    root.mainloop()
