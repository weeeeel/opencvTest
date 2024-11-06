import cv2
import csv
import mediapipe as mp
import time
import numpy as np
import itertools
import copy
import threading
import os
from pyparrot.Bebop import Bebop
from Model.Keypoint_classifier import KeyPointClassifier

# Threaded Video Capture class to ensure non-blocking camera input
class VideoStream:
    def __init__(self, src=0, width=320, height=240, fps=30):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# Cooldown Period between inputs
takeoff_cooldown = 1.0  # Reduced to 1 second for faster response
last_takeoff_time = 0
is_flying = False  # Track if the drone is currently flying
current_altitude = 0  # Track current altitude
landing_start_time = 0 #Track the time when landing is initiated 

# Define landing range (in cm)
landing_range = 50  # Change this value based on your needs
PrevHandSign = -1


  # Read labels ###########################################################
with open('Model/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
        ]

# Functions for landmark processing, bounding boxes, and gestures --------------
def Get_number_pressed(key, Training):

    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 116:  # t
        Training =  not Training
    return number,  Training

def Initialize_drone ():
    # Initialize drone connection and state
    bebop = Bebop()
    print("Connecting")

    if bebop.connect(10):
        print("Success")
    else: 
        print ("Failure")

    print("Sleeping")
    bebop.smart_sleep(5)
    bebop.ask_for_state_update()
    return bebop

def logging_csv(number, landmark_list):
    
    if  (0 <= number <= 9):
        csv_path = 'Model/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])

def make_landmark_list(image, landmarks):
    width, height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_X = min(int(landmark.x * width), width - 1)
        landmark_Y = min(int(landmark.y * height), height - 1)
        landmark_point.append([landmark_X, landmark_Y])

    return landmark_point

def normalize_Landmarks(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]

    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return (n / max_value)

    return list(map(normalize_, temp_landmark_list))

def make_bounding_rect(image, landmarks):
    webcam_width, webcam_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_X = min(int(landmark.x * webcam_width), webcam_width - 1)
        landmark_Y = min(int(landmark.y * webcam_height), webcam_height - 1)
        landmark_point = [np.array((landmark_X, landmark_Y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def draw_bounding_rect(image, brect, Training):
    if brect:
        if Training:
            cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 1)
        else:
            cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (255, 0, 0), 1)

def side(processed_landmark_list, classifier, bebop, current_time, Training):
    hand_sign_id = classifier(processed_landmark_list)
   
    # Control the drone based on the gesture
    if not Training:
        control_drone_with_gesture(hand_sign_id, bebop, current_time)

def control_drone_with_gesture(hand_sign_id, bebop, current_time):
    global last_takeoff_time, is_flying, current_altitude

    if hand_sign_id == 0:  # Open Hand - Takeoff or Ascend
        if not is_flying:  # Takeoff only if not already flying
            if current_time - last_takeoff_time > takeoff_cooldown:
                print('Open Hand - Takeoff')
                bebop.safe_takeoff(10)
                last_takeoff_time = current_time
                is_flying = True  # Set flying status
                current_altitude = 0  # Reset altitude on takeoff
        else:
            print("Open Hand - Ascending")
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=50, duration=1)  # Ascend
            current_altitude += 50  # Increment altitude

    elif hand_sign_id == 1:  # Closed Hand - Descend
        if is_flying:
            print('Closed Hand - Descending')
            bebop.fly_direct(roll=0, pitch=0, yaw=0, vertical_movement=-50, duration=1)  # Descend
            current_altitude -= 50  # Decrement altitude

            # Check if within landing range to execute landing
            if current_altitude <= landing_range:
                print('Landing command executed')
                bebop.safe_land(10)
                landing_time = current_time #record landing start time
                current_altitude = 0  # Reset altitude
                is_flying = False  # Update flying status

    elif hand_sign_id == 2:  # Pointing Gesture - Forward
        if is_flying:
            if current_time - last_takeoff_time > takeoff_cooldown:
                print ("Pointing - Going Forward")
                bebop.fly_direct(roll =0, pitch=50, yaw = 0, vertical_movement=0, duration=1) #Forward
                last_takeoff_time = current_time

    elif hand_sign_id == 3: # Ok Gesture - Backwards
        if is_flying:
            if current_time - last_takeoff_time > takeoff_cooldown:
                print ("Ok - Going Backwards")
                bebop.fly_direct(roll = 0, pitch= -50, yaw=0, vertical_movement=0, duration=1) #Backwards

   



# def side(processed_landmark_list, classifier, bebop, current_time):
#     hand_sign_id = classifier(processed_landmark_list)
    
#     # Control the drone based on the gesture
#     control_drone_with_gesture(hand_sign_id, bebop, current_time)

def main():
    drone = Initialize_drone()
    classifier = KeyPointClassifier()
    Training = False

    webcam = VideoStream(width=640, height=480, fps=15).start()

    # MediaPipe hands setup
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    processing_interval = 0.01  # Process gestures every 10ms
    last_processed_time = 0

    # while True:
    #     pass
    running = True
    while running:
        img = webcam.read()
        key = cv2.waitKey(10)

        if key == 27:  # ESC to quit
            running = False
            break
        
        number, Training = Get_number_pressed(key, Training)


        current_time = time.time()
        if current_time - last_processed_time > processing_interval:
            last_processed_time = current_time

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = make_landmark_list(img, hand_landmarks)
                    processed_landmark_list = normalize_Landmarks(landmark_list)

                    brect = make_bounding_rect(img, hand_landmarks)
                    draw_bounding_rect(img, brect, Training)

                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if Training:
                        logging_csv(number, processed_landmark_list )
                
                    threading.Thread(target=side, args=(processed_landmark_list, classifier, drone, current_time, Training)).start()

        # if not is_flying and (current_time - landing_start_time >= 5):
        #     print ("The drone has landed for 5 seconds. Disconnecting")
        #     bebop.disconnect()
        #     break

        cv2.imshow('Drone Control', img)

        if cv2.waitKey(1) == ord('q'):
            drone.disconnect(10)
            break

       

    webcam.stop()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == '__main__':
    main()