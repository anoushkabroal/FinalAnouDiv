import threading

import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Timer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

rightFootIndex, leftFootIndex, rightHeel, leftHeel, leftIndex, leftShoulder, rightIndex, rightShoulder = [], [], [], [], [], [], [], []
leftElbowAngle, rightElbowAngle, leftWristAngle, rightWristAngle, leftWrist, rightWrist, leftHip, rightHip, leftKnee, rightKnee, nose = 0, 0, 0, 0, [], [], [], [], [], [], []




image_path = ' '

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle #keep angle between 0-180

    return angle

vid = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose  # pose information getting
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils  # to draw on certain parts of the body
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


red_dot = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)  # draw red dots
green_line = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)

def check_landmarks_not_empty(*landmarks):
    return all(landmarks)

def doForAll(frame, remaining_time, dance_position):
    cv2.putText(frame, f'Time left: {int(remaining_time)}s', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Please demonstrate: ' + dance_position, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

def scoreFirst():
    if check_landmarks_not_empty(rightFootIndex, leftFootIndex, rightHeel, leftHeel, leftIndex, leftShoulder, rightIndex, rightShoulder):
        if (rightFootIndex[0] < rightHeel[0] and leftFootIndex[0] > leftHeel[0] and leftIndex[1] > leftShoulder[1] and
            rightIndex[1] > rightShoulder[1]):
            if (135 < leftElbowAngle < 165 and 135 < rightElbowAngle < 165 and 20 < leftWristAngle < 35 and 20 < rightWristAngle < 35):
                print("YOOOOO")

def scoreSecond():
    if check_landmarks_not_empty(rightFootIndex, leftFootIndex, rightHeel, leftHeel, leftIndex, leftShoulder, rightIndex, rightShoulder):
        if (rightFootIndex[0] < rightHeel[0] and leftFootIndex[0] > leftHeel[0] and leftFootIndex[0] > leftHip[0] and
            rightFootIndex[0] < rightHip[0] and leftWrist[0] > leftShoulder[0] and rightWrist[0] < rightShoulder[0]
            and nose[1] < leftWrist[1] < leftHip[1] and nose[1] < rightWrist[1] < rightHip[1]):
            if (145 < leftElbowAngle < 180 and 145 < rightElbowAngle < 180 and 0 < leftWristAngle < 10 and 0 < rightWristAngle < 10):
                print("YOOOOO")

def scoreFourth():
    if check_landmarks_not_empty(rightFootIndex, rightHeel, leftWrist, rightShoulder, leftShoulder, nose, leftHip):
        if (rightFootIndex[0] < rightHeel[0]):
            if (rightWrist[1] < nose[1] and rightShoulder[0] < leftWrist[0] < leftShoulder[0] and nose[1] < leftWrist[1] < leftHip[1]):
                if (0 < rightWristAngle < 20):
                    print("YOOOOO")

def scoreFifth():
    if check_landmarks_not_empty(rightFootIndex, leftFootIndex, rightHeel, leftKnee, leftWrist, nose, rightWrist):
        if (rightFootIndex[0] < rightHeel[0] or leftFootIndex[0] > leftHeel[0] and rightHeel[1] > leftKnee[1]):
            if (leftWrist[1] < nose[1] and rightWrist[1] < nose[1]):
                if (125 < leftElbowAngle < 160 and 125 < rightElbowAngle < 160 and 0 < rightWristAngle < 20 and 0 < leftWristAngle < 20):
                    print("YOOOOO")

def scorePasse():
    if check_landmarks_not_empty(leftFootIndex, leftHeel, rightKnee, rightHip, rightHeel, rightFootIndex, leftWrist, nose, rightShoulder, leftShoulder, rightWrist, leftHip):
        if (leftFootIndex[0] > leftHeel[0] and rightKnee[0] < rightHip[0] and rightHeel[1] < leftKnee[1] and rightHeel[0] < rightFootIndex[0]):
            if (leftWrist[1] < nose[1] and rightWrist[1] < nose[1]):
                print("YOOOOO")
        elif (leftFootIndex[0] > leftHeel[0]):
            if (leftWrist[1] < nose[1] and rightShoulder[0] < rightWrist[0] < leftShoulder[0] and nose[1] < rightWrist[1] < rightHip[1]):
                if (0 < leftWristAngle < 20):
                    print("YOOOOO")

def totalScore(score1, score2, score3, score4, score5):
    return score1 + score2 + score3 + score4 + score5

def startingScreen():
    print("Nothing")
def start_timer():
    threading.Timer(0, startingScreen).start()
    threading.Timer(30, scoreFirst).start()
    threading.Timer(60, scoreSecond).start()
    threading.Timer(90, scoreFourth).start()
    threading.Timer(120, scoreFifth).start()
    threading.Timer(150, scorePasse).start()
    threading.Timer(180, totalScore(1,1,1,1,1)).start()

start_time = time.time()
timer_duration = 30  # Duration for each position in seconds

start_timer()
while True:

    # Capture the video frame by frame
    ret, frame = vid.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640,640))

    results = pose.process(frame)  # result
    #image = cv2.putText(frame, 'Your pose is: ', (50, 50), 4, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # red landmarks, and green connection lines
    height, width, _ = frame.shape
    if (results.pose_landmarks):
        # wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height
        # nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.CHIN].y * height

        nose = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y]

        leftWrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        leftIndex = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX.value].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
        leftPinky = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY.value].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY.value].y]
        leftElbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        leftShoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        leftFootIndex = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
        leftHeel = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
        leftHip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        leftKnee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        rightWrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        rightIndex = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
        rightPinky = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY.value].y]
        rightElbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        rightShoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        rightFootIndex = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
        rightHeel = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
        rightHip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        rightKnee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                   results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

       # nose = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].x,
         #             results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y]

        leftWristAngle = calculate_angle(leftIndex, leftWrist, leftPinky)
        rightWristAngle = calculate_angle(rightIndex, rightWrist, rightPinky)
        leftElbowAngle = calculate_angle(leftWrist, leftElbow, leftShoulder)
        rightElbowAngle = calculate_angle(rightWrist, rightElbow, rightShoulder)
        heelAngle  = calculate_angle(rightFootIndex, leftHeel, leftFootIndex)
        rightKneeAngle = calculate_angle(rightHeel, rightKnee, rightHip)
        #rightHeelAngle = calculate_angle(rightHip, rightHeel, rightFootIndex)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # fix color

        # multiply the position of the text so that it is next to left wrist times webcam dimensions
        cv2.putText(frame, str(round(leftWristAngle,1)),
                    tuple(np.multiply(leftWrist, [640, 600]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

        cv2.putText(frame, str(round(rightWristAngle,1)),
                    tuple(np.multiply(rightWrist, [500, 600]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )
        cv2.putText(frame, str(round(leftElbowAngle, 1)),
                    tuple(np.multiply(leftElbow, [640,600]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )
        cv2.putText(frame, str(round(rightElbowAngle,1)),
                    tuple(np.multiply(rightElbow, [500, 600]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

        cv2.putText(frame, str(round(heelAngle, 1)),
                    tuple(np.multiply(leftHeel, [670, 640]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )


        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=red_dot,
                               connection_drawing_spec=green_line)

        #timer_duration = 30  #how long to get into each pose


        #current time

       # t = Timer(30, scoreFirst)
       # t.start()
        #while t.is_alive():
       #     image = cv2.putText(frame, 'Please demonstrate First Position', (20, 50), 4, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Display timer on the frame
        elapsed_time = time.time() - start_time
        remaining_time = timer_duration - (elapsed_time % timer_duration)
        position_phase = int(elapsed_time // timer_duration) + 1
        dance_position = " "
        if(position_phase == 1):
            cv2.putText(frame, 'Welcome to Ballet Bonanza!', (60, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Get ready! In 30 seconds you', (94, 570),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'should be prepared with an open space to', (10, 590),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'pose', (310, 610),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
        elif(position_phase == 2):
            dance_position = "First Position"
            image_path = 'firstPosition.png'
            doForAll(frame,remaining_time, dance_position)
            cv2.imshow('frame', frame)
            overlay_image = cv2.imread(image_path)
            overlay_image = cv2.resize(overlay_image, (200, 400))
            cv2.imshow('Example', overlay_image)
        elif(position_phase == 3):
            dance_position = "Second Position"
            image_path = 'secondPosition.png'
            doForAll(frame, remaining_time, dance_position)
            cv2.imshow('frame', frame)
            overlay_image = cv2.imread(image_path)
            overlay_image = cv2.resize(overlay_image, (200, 400))
            cv2.imshow('Example', overlay_image)
        elif (position_phase == 4):
            dance_position = "Fourth Position"
            image_path = 'fourthPosition.png'
            doForAll(frame, remaining_time, dance_position)
            cv2.imshow('frame', frame)
            overlay_image = cv2.imread(image_path)
            overlay_image = cv2.resize(overlay_image, (200, 400))
            cv2.imshow('Example', overlay_image)
        elif (position_phase == 5):
            dance_position = "Fifth Position"
            image_path = 'fifthPosition.png'
            doForAll(frame, remaining_time, dance_position)
            cv2.imshow('frame', frame)
            overlay_image = cv2.imread(image_path)
            overlay_image = cv2.resize(overlay_image, (200, 400))
            cv2.imshow('Example', overlay_image)
        elif (position_phase == 6):
            dance_position = "Right Passe"
            image_path = 'rightPasse.png'
            doForAll(frame, remaining_time, dance_position)
            cv2.imshow('frame', frame)
            overlay_image = cv2.imread(image_path)
            overlay_image = cv2.resize(overlay_image, (200, 400))
            cv2.imshow('Example', overlay_image)
        elif(position_phase == 7):
            print("The score should be printed here and also maybe pass in all the previous scores somehow")
            cv2.putText(frame, 'Your cumulative score is: (score)', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            image_path = 'goodJob.jpg'
            cv2.imshow('frame', frame)
            overlay_image = cv2.imread(image_path)
            overlay_image = cv2.resize(overlay_image, (200, 400))
            cv2.imshow('Slay!', overlay_image)
            #cv2.destroyWindow('Example')
        else: #END IT ALL! Break doesn't work though
            cv2.destroyAllWindows()
            break
            #break


        '''
        start_time = time.time()

        #Start checks here
       # image = cv2.putText(frame, 'Time (30 seconds till score: '+t, (300, 50), 4, 1, (0, 0, 0), 2, cv2.LINE_AA)
        while time.time() - start_time < timer_duration:
            image = cv2.putText(frame, 'Please demonstrate First Position', (20, 50), 4, 1, (0, 0, 0), 2, cv2.LINE_AA)
        scoreFirst()
        #Then display scoreFirst for a few seconds and also save it to a total
        '''

    # q key is quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Post release the cap object
vid.release()
# vid2.release()


