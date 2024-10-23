import cv2
import mediapipe as mp
import time
import numpy as np
import itertools
import csv
import copy


from Model.Keypoint_classifier import KeyPointClassifier

#functions------------------------------------------------------------------------------------------------------------------------------------

def make_landmark_list(image, landmarks):

    width , height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):

        landmark_X = min(int(landmark.x * width), width - 1)
        landmark_Y = min(int(landmark.y * height), height - 1)
        landmark_point.append([landmark_X, landmark_Y])

    return landmark_point

def normalize_Landmarks(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0

    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y  = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # convert landmark list into 1d list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # normalize
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
            return (n / max_value)
    
    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, landmark_list):
    
    if (0 <= number <= 9):
        csv_path = 'Model/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def make_bounding_rect(image, landmarks):
    webcam_width, webcam_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0,2), int)

    for _, landmark in enumerate(landmarks.landmark):

        landmark_X = min(int(landmark.x * webcam_width), webcam_width - 1)
        landmark_Y = min(int(landmark.y * webcam_height), webcam_height - 1)
        
        landmark_point = [np.array((landmark_X, landmark_Y))]
        landmark_array = np.append(landmark_array, landmark_point, axis = 0)
    
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

def draw_bounding_rect(use_brect, image, brect, training):
    if use_brect:
        if training:
            cv2.rectangle(image, (brect[0], brect[1]),(brect[2], brect[3]), (0, 255, 0), 1 )
        else:
            cv2.rectangle(image, (brect[0], brect[1]),(brect[2], brect[3]), (0, 0, 255), 1 )

def set_the_throttle(input):
    return (input - 0.5)* 100

def main():

    # set dt variables
    timeBetweenDt= 2 # displays the frame rate every 2 seconds
    dt = 0
    frameCounter = 0
    start_time = time.time()


    # crate a hand geasture classifier object
    Classifier = KeyPointClassifier()
    training = False

    #loop every frame -------------------------------------------------------------------------------------------------------------------------
    while True:
        success, img = webcam.read()
        key = cv2.waitKey(10)

       
        if key == 27:  # ESC
            break
        elif key == 116:
            training = not training
        
        # Convert image to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        #/key = keyboard.read_key()
    
        #if  key =='t':
        #    training = not training
       # else:
        #    pass

    
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

    
                #get the landmark data 
                landmark_list = make_landmark_list(img, hand_landmarks)
                #normalise the landmark data 
                processed_landmark_list = normalize_Landmarks(landmark_list)

                #calculate bounding rectangle
                brect = make_bounding_rect(img, hand_landmarks)
                
                #draw landmarks
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                #draw bounding
                img_rgb = draw_bounding_rect(True, img, brect, training)

                # hand sign classification

                hand_sign_id = Classifier(processed_landmark_list)
                match hand_sign_id:
                                    case 0:
                                        print('open')
                                    case 1:
                                        print('close')
                                    case 2:
                                        print('point')
                                    case 3:
                                        print('ok')
                #print(processed_landmark_list)

                #print("\n")
                
        

        
        # output the average time between frames in ms
        cv2.putText(img,  f'ms: {1000*dt:.2f}',  (550, 25), cv2.FONT_HERSHEY_PLAIN , 1, (255, 255, 255),  1,) 
        
        # Show the processed image with hand annotations 
        cv2.imshow('Artemis', img)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            return
        
        # every x amount if time show the average dt between frames 
        frameCounter+=1
        if (time.time() - start_time) > timeBetweenDt :
            dt = (time.time() - start_time)/frameCounter
            frameCounter = 0
            start_time = time.time()

    # Release the webcam and close all OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()

#------------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hands model outside the loop to improve efficiency
hands = mp_hands.Hands(max_num_hands=1)
webcam = cv2.VideoCapture(0)


main()