import cv2
import pyvirtualcam
import time
import mediapipe as mp
from threading import Lock, Thread
import PySimpleGUI as sg


# Webcam switcher based on face-detection
# By group 3 - Robin Witmer, Alexandru Malgras and Jelle Schroijen
# Last edit - 17/09/2021

# ------------------------------------------
#                 Parameters
# ------------------------------------------

# Output cam
output_size_width = 1080
output_size_height = 720
output_fps = 20
output_switch_delay = 1000

# Camera names
output_camera_name = False
preview_camera_names = True
camera_name_font_family = cv2.FONT_HERSHEY_PLAIN
camera_name_font_scale = 1
camera_name_font_color = (255, 255, 255) # BGR

# Preview window
preview_enabled = True
preview_draw_rectangles = True
preview_rectangle_color = (0, 0, 255)  # BGR
preview_highlight_selected = True
preview_highlight_color = (0, 255, 0) # BGR

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
standing = False

refPt = []
# ------------------------------------------
#                  Program
# ------------------------------------------

# Initialization
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
input_1 = cv2.VideoCapture(0) # cv2.VideoCapture('D:\\Users\\Jelle\\Videos\\scenario1\\cam_video.mp4')
input_2 = cv2.VideoCapture(1)
standingTIme = 0
sittingTIme = 0
lastDetected = 0
framecount = 0

image = None
standing = False
stop = False

#initiate the GUI
genericY = 0
heigthValueCalculating = 0.4

def detectPose():
    global image, standing, stop, lastDetected
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, enable_segmentation=True) as holistic:
        while not stop:
            results = holistic.process(image)
            if results.pose_landmarks != None :
                lastDetected = int(time.time() * 1000.0)
                #print(lastDetected)
                print(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)
                if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].visibility >0.9 and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].visibility  >0.9 and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y <heigthValueCalculating and results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y <heigthValueCalculating :
                    #print('Standing')    
                    standing = True
                else:
                    standing = False
                    
detector = Thread(target = detectPose)

def calculate_height(yInput):
  global heigthValueCalculating
  heigthValueCalculating = yInput /frames_combined.shape[0] 
  print('heigth = ' + str(heigthValueCalculating))
  

def click_and_select(event, x, y, flags, param):
	# grab references to the global variables
	global refPt
	global genericY
	if event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		
		genericY = y
		calculate_height(y)
		# draw a rectangle around the region of interest
		cv2.line(frames_combined, pt1=(0,y), pt2=(4000,y), color=(0,0,255), thickness=10)
		cv2.imshow("frames_combined", frames_combined)
	else:
		cv2.line(frames_combined, pt1=(0,genericY), pt2=(4000, genericY), color=(0,0,255), thickness=10)
		cv2.imshow("frames_combined", frames_combined)

        




with pyvirtualcam.Camera(width=output_size_width, height=output_size_height, fps=output_fps) as output:
    print(f'Using output device: {output.device}')

    # Main loop
    while True:
        # Capture frame-by-frame
        ret_1, frame_1 = input_1.read()
        ret_2, frame_2 = input_2.read()
        
        # Try to detect faces
        grayscale = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY) # Convert into grayscale
        faces = face_cascade.detectMultiScale(grayscale, 1.1, 4) # Apply cascade
        image = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
        if not detector.is_alive():
            detector.start()
                
        # Set the output frame
        output_frame = cv2.resize(frame_2, (output_size_width, output_size_height))
        output_camera = 'Camera 2'
        time_now_ms = int(time.time() * 1000.0)
        if standing:
            standingTIme = time_now_ms
        else :
            sittingTIme = time_now_ms
        
        if standingTIme >= (time_now_ms - output_switch_delay):
            output_frame = cv2.resize(frame_1, (output_size_width, output_size_height))
            output_camera = 'Camera 1'
        if sittingTIme >= (time_now_ms - output_switch_delay):
            output_frame = cv2.resize(frame_2, (output_size_width, output_size_height))
            output_camera = 'Camera 2'
        if lastDetected < (time_now_ms - 5000):
            output_frame = cv2.resize(frame_2, (output_size_width, output_size_height))
            output_camera = 'Camera 2'
        # Show camera name
        if (output_camera_name):
            output_frame = cv2.putText(output_frame, output_camera, (50, 50), camera_name_font_family, camera_name_font_scale, camera_name_font_color, 1, cv2.LINE_AA)

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB) # Convert to RGB
        output.send(output_frame) # Display the output

        # Preview window
        if (preview_enabled):
            # Resize to match
            max_height = max(img.shape[0] for img in [frame_1, frame_2])
            frame_1 = cv2.resize(frame_1, (int(frame_1.shape[1] * (max_height / frame_1.shape[0])), max_height))
            frame_2 = cv2.resize(frame_2, (int(frame_2.shape[1] * (max_height / frame_2.shape[0])), max_height))

            # Draw rectangle around the faces
            if (preview_draw_rectangles):
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_1, (x, y), (x+w, y+h), preview_rectangle_color, 2)
            
            # Show camera names
            if (preview_camera_names):
                frame_1 = cv2.putText(frame_1, "Camera 1", (50, 50), camera_name_font_family, camera_name_font_scale, camera_name_font_color, 1, cv2.LINE_AA)
                frame_2 = cv2.putText(frame_2, "Camera 2", (50, 50), camera_name_font_family, camera_name_font_scale, camera_name_font_color, 1, cv2.LINE_AA)

            # Show selected
            if (preview_highlight_selected):
                if standingTIme >= (time_now_ms - output_switch_delay):
                    frame_1 = cv2.copyMakeBorder(frame_1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=preview_highlight_color)
                    frame_2 = cv2.copyMakeBorder(frame_2, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                else:
                    frame_1 = cv2.copyMakeBorder(frame_1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    frame_2 = cv2.copyMakeBorder(frame_2, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=preview_highlight_color)
                if lastDetected < (time_now_ms - 5000):
                    frame_1 = cv2.copyMakeBorder(frame_1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    frame_2 = cv2.copyMakeBorder(frame_2, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=preview_highlight_color)
            else:
                frame_1 = cv2.copyMakeBorder(frame_1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                frame_2 = cv2.copyMakeBorder(frame_2, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            # Combine
            frames_combined = cv2.hconcat([frame_1, frame_2])

            # Show preview
            cv2.namedWindow("frames_combined")
            cv2.setMouseCallback("frames_combined", click_and_select)
            cv2.imshow('frames_combined', frames_combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        # Wait for next frame
       
        output.sleep_until_next_frame()

# When everything is done, close related windows
input_1.release()
input_2.release()
stop = True
cv2.destroyAllWindows()
