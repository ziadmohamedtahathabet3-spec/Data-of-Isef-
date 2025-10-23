import os
import datetime
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import pickle
import util
from ZI import test  # Assuming test function is defined in test.py

# import tensorflow as tf
# from tensorflow.keras.models import load_model

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")
        self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle main window closing

        self.login_button_main_window = util.get_button(self.main_window, 'تسجيل', 'green', self.login)
        self.login_button_main_window.place(x=400, y=50)

        # Load and display the image
        self.load_logo()

        # Initialize necessary attributes
        self.db_dir = './id_card'
        self.parson = "D:/Updating_data/assits/image/faces"
        self.log_path = './log.txt'
        
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        if not os.path.exists(self.parson):
            os.mkdir(self.parson)

    def load_logo(self):
        path_logo = "D:/Updating_data/assits/image/update_data.jpg"  # Replace with your logo path
        image = Image.open(path_logo)
        image = image.resize((300, 300), Image.Resampling.LANCZOS)  # Resize the image to fit in the window
        self.logo_image = ImageTk.PhotoImage(image)

        self.logo_label = tk.Label(self.main_window, image=self.logo_image)
        self.logo_label.place(x=400, y=200)

    def login(self):
        self.next_window = tk.Toplevel(self.main_window)
        self.next_window.geometry("1200x520+350+100")
        self.next_window.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle next window closing

        next_window_user_name_label = util.get_text_label(self.next_window, 'Please, input your username:')
        next_window_user_name_label.place(x=450, y=70)

        self.next_window_user_name_enter = tk.Entry(self.next_window)
        self.next_window_user_name_enter.place(x=450, y=150, width=300, height=30)

        next_window_password_label = util.get_text_label(self.next_window, 'Please, input your password:')
        next_window_password_label.place(x=450, y=230)

        self.next_window_password_enter = tk.Entry(self.next_window, show="*")
        self.next_window_password_enter.place(x=450, y=310, width=300, height=30)

        self.th_button_next_window = util.get_button(self.next_window, 'انتقال', 'gray', self.th, fg='green')
        self.th_button_next_window.place(x=450, y=370, width=300, height=30)

        self.main_window.withdraw()  # Hide the main window

    def th(self):
        dec_user_pass = {
            'Mohamed Waleed': 'J0223045',
            'Ziad Mohamed': 'J0223042',
            'Salsabil Mostafa': 'J0223031',
            'Sherry Sheref': 'J0223034',
            'Youssef Abdel Aziz': 'J0223050'
        }

        user_name = self.next_window_user_name_enter.get()
        password = self.next_window_password_enter.get()

        if user_name in dec_user_pass and dec_user_pass[user_name] == password:
            self.user_save = user_name
            self.therd_page()
            self.next_window.withdraw()
        else:
            util.msg_box('error', 'Make sure your username and password are correct')

    def therd_page(self):
        self.therd_page_window = tk.Toplevel(self.next_window)
        self.therd_page_window.geometry("1200x520+350+100")
        self.therd_page_window.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle third page closing

        self.webcam_label = util.get_img_label(self.therd_page_window)
        self.webcam_label.place(x=10, y=1, width=1000, height=500)
        self.add_webcam(self.webcam_label)

        self.take_id_card_button_therd_page = util.get_button(self.therd_page_window, 'التقاط وجه البطاقة', 'blue', self.take_id_card, fg='black')
        self.take_id_card_button_therd_page.place(x=800, y=300)

        self.take_back_id_card_button_therd_page = util.get_button(self.therd_page_window, 'التقاط ظهر البطاقة', 'blue', self.back_id_card, fg='black')
        self.take_back_id_card_button_therd_page.place(x=800, y=400)

    def load_labels(self, labels_path):
        with open(labels_path, 'r') as f:
            labels = f.read().strip().split('\n')
        return labels

    def load_model(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()

        if not ret:
            util.msg_box('Webcam Error', 'Failed to capture image from webcam. Please check your camera and try again.')
            return

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def take_id_card(self):
        self.id_card_img = self.most_recent_capture_arr.copy()
        cv2.imwrite(os.path.join(self.db_dir, "id_card_front.jpg"), self.id_card_img)

        # Display a message box with the prediction result
        util.msg_box('successfully!', 'The image has been saved')

    def back_id_card(self):
        self.back_picture = self.most_recent_capture_arr.copy()
        cv2.imwrite(os.path.join(self.db_dir, "id_card_back.jpg"), self.back_picture)
        util.msg_box('successfully!', f'operation accomplished successfully {self.user_save}')
        self.forth_page()

    def forth_page(self):
        self.forth_page_window = tk.Toplevel(self.therd_page_window)
        self.forth_page_window.geometry("1200x520+350+100")
        self.forth_page_window.protocol("WM_DELETE_WINDOW", self.on_closing)  # Handle forth page closing

        self.save_button_forth_page = util.get_button(self.forth_page_window, 'حفظ', 'yellow', self.save_and_operate, fg='purple')
        self.save_button_forth_page.place(x=750, y=200)

        self.webcam_label_2 = util.get_img_label(self.forth_page_window)
        self.webcam_label_2.place(x=10, y=0, width=700, height=500)

        self.add_webcam_2(self.webcam_label_2)
        self.therd_page_window.withdraw()

    def add_webcam_2(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        self._label_2 = label
        self.process_webcam_2()

    def process_webcam_2(self):
        ret, frame = self.cap.read()

        if not ret:
            util.msg_box('Webcam Error', 'Failed to capture image from webcam. Please check your camera and try again.')
            return

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label_2.imgtk = imgtk
        self._label_2.configure(image=imgtk)

        self._label_2.after(20, self.process_webcam_2)

    def save_and_operate(self):
        # Load image from webcam
        webcam_image = self.most_recent_capture_arr
        
        # Check if the webcam image is valid
        if webcam_image is None:
            util.msg_box('Error', 'No image captured from webcam. Please try again.')
            return
        
        # Load face encodings from the 'self.parson' directory
        known_face_encodings = []
        known_face_names = []
        for filename in os.listdir(self.parson):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.parson, filename)
                person_image = face_recognition.load_image_file(image_path)
                person_face_encoding = face_recognition.face_encodings(person_image)[0]  # Assuming there's only one face in each image
                known_face_encodings.append(person_face_encoding)
                known_face_names.append(os.path.splitext(filename)[0])  # Extract the name without extension

        # Find face locations and encodings in the webcam image
        face_locations = face_recognition.face_locations(webcam_image)
        face_encodings = face_recognition.face_encodings(webcam_image, face_locations)

        for face_encoding in face_encodings:
            # Compare face encoding with known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"
            label = test(
                image=webcam_image,
                model_dir="D:/Silent-Face-Anti-Spoofing-master (1)/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models",
                device_id=0
            )

            if label == 1:
                # Check if any face matches
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    break

        # Perform actions based on the result
        if name != "Unknown":
            util.msg_box('Welcome back!', 'Welcome, {}.'.format(name))
            with open(self.log_path, 'a') as f:
                f.write('{},{},in\n'.format(name, datetime.datetime.now()))
        elif label != 1:
            util.msg_box('Hey, you are a spoofer!', 'You are fake!')
        else:
            util.msg_box('Unknown user', 'Unknown user. Please register new user or try again.')


    def on_closing(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.main_window.destroy()

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()
