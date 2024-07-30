import threading

import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import messagebox
from threading import Thread
import time

def show_message():
    root = tk.Tk()
    root.withdraw()  # Pencereyi gizle
    messagebox.showinfo("Bilinmeyen Kisi!", "Bilinmeyen kisi!")
    root.after(3000, root.destroy)  # 3 saniye sonra pencereyi kapat
    root.mainloop()

def show_warning():
    # 3 saniye boyunca uyarı penceresi gösterme
    cv2.namedWindow('Warning', cv2.WINDOW_NORMAL)
    warning_message = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(warning_message, "Bilinmeyen kisi!", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Warning', warning_message)
    cv2.waitKey(3000)
    cv2.destroyWindow('Warning')

# Yüz algılama ve işaret noktaları tespiti için dlib yükleme
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Yüz tanıma modelini yükleme
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)

# Kamera açma
cap = cv2.VideoCapture(0)

# Kişilere ait tanımlayıcı özellikleri saklamak için boş bir liste oluşturma
known_face_encodings = []
known_face_names = []


# Kendi yüzünüzü kaydetmek için örnek
def register_known_face(face_image_path, name):
    # Yüzü oku
    image = cv2.imread(face_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Yüzleri algıla
    faces = detector(gray)
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray, face)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(image, landmarks))
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)


# Örnek olarak kendinizin yüzünü kaydedebilirsiniz
register_known_face('faces/Hasan.jpg', 'Hasan')
register_known_face('faces/Fatih.jpg', 'Fatih')
register_known_face('faces/Eren.jpg', 'Eren')

while True:
    # Kameradan görüntü okuma
    ret, frame = cap.read()
    if not ret:
        break

    # Gri tonlamalı görüntüye çevirme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri algılama
    faces = detector(gray)

    for face in faces:
        # Yüz çevresine dikdörtgen çizme
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # İşaret noktalarını tespit etme
        landmarks = predictor(gray, face)
        face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, landmarks))

        # Tanıma işlemi
        distances = [np.linalg.norm(face_encoding - known_encoding) for known_encoding in known_face_encodings]
        if distances:
            min_distance = min(distances)
            if min_distance < 0.6:  # Threshold, daha hassas ayarlayabilirsiniz
                name = known_face_names[distances.index(min_distance)]
            else:
                name = "Unknown"
                threading.Thread(target=show_warning).start()
        else:
            name = "Unknown"
            threading.Thread(target=show_warning).start()

        # Yüz adını yazdırma
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Görüntüyü gösterme
    cv2.imshow('Face Recognition', frame)

    # 'q' tuşuna basıldığında döngüyü sonlandırma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve pencereleri kapatma
cap.release()
cv2.destroyAllWindows()
