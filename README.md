# Face Recognition with Real-Time Alerts

## Project Overview

This project implements a real-time face recognition system using OpenCV and dlib. It detects known faces and provides alerts if an unknown face is recognized. The system uses both graphical and textual notifications to indicate unknown individuals.

## Features

- Real-time face detection and recognition.
- Alerts for unknown individuals via pop-up message and on-screen warning.
- Multi-threaded approach to handle alerts separately from the main face recognition process.

## Dependencies

- OpenCV
- dlib
- NumPy
- Tkinter (for GUI alerts)

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository

2. **Install Dependencies:**
Ensure you have Python installed, then install the required packages:

bash
```
pip install opencv-python dlib numpy
```

Download dlib Models:

Download shape_predictor_68_face_landmarks.dat from dlib's model repository.
Download dlib_face_recognition_resnet_model_v1.dat from dlib's model repository.
Place these files in the same directory as your script or update the paths accordingly in the script.

2. **Add Known Faces:**

Place images of known individuals in the faces directory.
Update the register_known_face function calls with the paths to your images and names.

example
bash
```
register_known_face('faces/Hasan.jpg', 'Hasan')
register_known_face('faces/Fatih.jpg', 'Fatih')
register_known_face('faces/Eren.jpg', 'Eren')
```

Run the Script:
Execute the Python script to start the face recognition system:

bash
```
python face_recognition.py
```
### Interaction:

The system will display a window with the camera feed.
It will draw rectangles around detected faces and display the recognized name above the face.
If an unknown face is detected, a warning window will pop up, and an on-screen warning will be displayed for 3 seconds.

### Stopping the Program:

Press the 'q' key in the video window to stop the program and close the camera feed.

### Face Detection and Recognition:

Uses dlib to detect faces and extract facial landmarks.
Computes face encodings and compares them to known faces.
Alerts are triggered for unknown faces.

### Alerts:

Graphical Alert: Uses Tkinter to show a pop-up message.
On-screen Warning: Uses OpenCV to display a warning message.
