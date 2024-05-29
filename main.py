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
score1 = 0
score2 = 0
score3 = 0
score4 = 0
score5 = 0
def doForAll(frame, remaining_time, dance_position):
    texts = [
        (f'Time left: {int(remaining_time)}s', (10, 30), 1),
        (f'Please demonstrate: ' + dance_position, (10, 80), 1)
    ]

    for text, (x, y), scale in texts:
        (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)

        l = (x, y - h - b)
        r = (x + w, y + b)
        cv2.rectangle(frame, l, r, (255, 255, 255), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)

def rectangles(texts):
    for text, (x, y), scale in texts:
        (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)

        l = (x, y - h - b)
        r = (x + w, y + b)
        cv2.rectangle(frame, l, r, (255, 255, 255), -1)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)

def scoreFirst():
    global score1
    if check_landmarks_not_empty(rightFootIndex, leftFootIndex, rightHeel, leftHeel, leftIndex, leftShoulder, rightIndex, rightShoulder):
        if (rightFootIndex[0] < rightHeel[0] and leftFootIndex[0] > leftHeel[0] and leftIndex[1] > leftShoulder[1] and
            rightIndex[1] > rightShoulder[1]):
            if 145 < leftElbowAngle < 155 and 145 < rightElbowAngle < 155 and 18 < leftWristAngle < 35 and 18 < rightWristAngle < 35:
                print("YOOOOO")
                score1 = 100
                texts = [
                    (f'Excellent!', (10, 540), 0.8)
                ]
                rectangles(texts)

            elif 125 < leftElbowAngle < 165 and 125 < rightElbowAngle < 165 and 8 < leftWristAngle < 55 and 8 < rightWristAngle < 55:
                score1 = 90
                texts = [
                    (f'Think about rounding your wrists!', (10, 540), 0.8),
                    (f'Try to get your wrist angles', (10, 580), 0.8),
                    (f'closer to 27', (10, 610), 0.8)
                ]
                rectangles(texts)
            elif 115 < leftElbowAngle < 175 and 115 < rightElbowAngle < 175 and 5 < leftWristAngle < 65 and 5 < rightWristAngle < 65:
                score1 = 70
                texts = [
                    (f'Think about rounding your elbows!', (10, 540), 0.8),
                    (f'Try to get your elbow angles', (10, 580), 0.8),
                    (f'closer to 150', (10, 610), 0.8)
                ]
                rectangles(texts)
            elif 105 < leftElbowAngle < 185 and 105 < rightElbowAngle < 185 and 5 < leftWristAngle < 75 and 5 < rightWristAngle < 75:
                score1 = 50
                texts = [
                    (f'Think about rounding your elbows!', (10, 540), 0.8),
                    (f'Try to get your elbow angles', (10, 580), 0.8),
                    (f'closer to 145', (10, 610), 0.8)
                ]
                rectangles(texts)
        else:
            score1 = 0
            print("not doing first position")
            texts = [
                (f'Please mirror the dancer example', (10, 540), 0.8),
            ]
            rectangles(texts)
        return score1
def scoreSecond():
    global score2
    if check_landmarks_not_empty(rightFootIndex, leftFootIndex, rightHeel, leftHeel, leftIndex, leftShoulder, rightIndex, rightShoulder):
        if (rightFootIndex[0] < rightHeel[0] and leftFootIndex[0] > leftHeel[0] and leftFootIndex[0] > leftHip[0] and
            rightFootIndex[0] < rightHip[0] and leftWrist[0] > leftShoulder[0] and rightWrist[0] < rightShoulder[0]
            and nose[1] < leftWrist[1] < leftHip[1] and nose[1] < rightWrist[1] < rightHip[1]):
            if (145 < leftElbowAngle < 170 and 145 < rightElbowAngle < 170 and 0 < leftWristAngle < 40 and 0 < rightWristAngle < 40):
                print("YOOOOO")
                score2 = 100
                texts = [
                    (f'Excellent!', (10, 540), 0.8),
                ]
                rectangles(texts)
            elif 135 < leftElbowAngle < 180 and 135 < rightElbowAngle < 180 and 0 < leftWristAngle < 40 and 0 < rightWristAngle < 40:
                score2 = 90
                texts = [
                    (f'Think about rounding your elbows DOWN!', (10, 540), 0.8),
                    (f'Try to get your elbow angles', (10, 580), 0.8),
                    (f'closer to 158', (10, 610), 0.8)
                ]
                rectangles(texts)
            elif 125 < leftElbowAngle < 190 and 125 < rightElbowAngle < 190 and 0 < leftWristAngle < 40 and 0 < rightWristAngle < 40:
                score2 = 70
                texts = [
                    (f'Think about rounding your elbows!', (10, 540), 0.8),
                    (f'Try to get your elbow angles', (10, 580), 0.8),
                    (f'closer to 140', (10, 610), 0.8)
                ]
                rectangles(texts)
            elif 100 < leftElbowAngle < 200 and 100 < rightElbowAngle < 200 and 0 < leftWristAngle < 60 and 0 < rightWristAngle < 60:
                score2 = 50
                texts = [
                    (f'Think about rounding your elbows and wrists!', (10, 540), 0.8),
                    (f'Try to get your elbow angles', (10, 580), 0.8),
                    (f'closer to 135', (10, 610), 0.8)
                ]
                rectangles(texts)
        else:
            score2 = 0
            print("not doing second position")
            texts = [
                (f'Please mirror the dancer shown', (10, 540), 0.8),
            ]
            rectangles(texts)
        return score2

def scoreFourth():
    global score3
    if check_landmarks_not_empty(rightFootIndex, rightHeel, leftWrist, rightShoulder, leftShoulder, nose, leftHip):
        if (rightFootIndex[0] < rightHeel[0]):
            if (rightWrist[1] < nose[1] and rightShoulder[0] < leftWrist[0] < leftShoulder[0] and nose[1] < leftWrist[1] < leftHip[1]):
                if (6 < rightWristAngle < 9 and 10< leftWristAngle < 30 and 120 < rightElbowAngle < 160  and 60 < leftElbowAngle < 90):
                    score3 = 100
                    print("YOOOOO")
                    texts = [
                        (f'Excellent!', (10, 540), 0.8),
                    ]
                    rectangles(texts)
                elif(5 < rightWristAngle < 10 and 5< leftWristAngle < 40 and 110 < rightElbowAngle < 170  and 55 < leftElbowAngle < 95):
                    score3 = 90
                    texts = [
                        (f'Think about rounding your wrists!', (10, 540), 0.8),
                        (f'Try to get your left wrist', (10, 580), 0.8),
                        (f'closer to 7', (10, 610), 0.8)
                    ]
                    rectangles(texts)
                elif (0 < rightWristAngle < 20 and 0 < leftWristAngle < 50 and 100 < rightElbowAngle < 180  and 50 < leftElbowAngle < 105):
                    score3 = 70
                    texts = [
                        (f'Think about rounding your elbows!', (10, 540), 0.8),
                        (f'Try to get your left elbow', (10, 580), 0.8),
                        (f'closer to 78', (10, 610), 0.8)
                    ]
                    rectangles(texts)
                elif (0 < rightWristAngle < 30 and 0 < leftWristAngle < 60 and 90 < rightElbowAngle < 190  and 40 < leftElbowAngle < 115):
                    score3 = 50
                    texts = [
                        (f'Think about putting your left arm lower!', (10, 540), 0.8),
                        (f'Try to get your right elbow', (10, 580), 0.8),
                        (f'closer to 140', (10, 610), 0.8)
                    ]
                    rectangles(texts)
            else:
                score3 = 0
                print("not doing fourth")
                texts = [
                    (f'Please mirror the dancer example', (10, 540), 0.8),
                ]
                rectangles(texts)
            return score3

def scoreFifth():
    global score4
    if check_landmarks_not_empty(rightFootIndex, leftFootIndex, rightHeel, leftKnee, leftWrist, nose, rightWrist):
        if (rightFootIndex[0] < rightHeel[0] or leftFootIndex[0] > leftHeel[0] and rightHeel[1] > leftKnee[1]):
            if (leftWrist[1] < nose[1] and rightWrist[1] < nose[1]):
                if (140 < leftElbowAngle < 160 and 140 < rightElbowAngle < 160 and 9 < rightWristAngle < 17 and 9 < leftWristAngle < 17):
                    print("YOOOOO")
                    score4 = 100
                    texts = [
                        (f'Excellent!', (10, 540), 0.8)
                    ]
                    rectangles(texts)
                elif (130 < leftElbowAngle < 170 and 130 < rightElbowAngle < 170 and 5 < rightWristAngle < 27 and 5 < leftWristAngle < 27):
                    score4 = 90
                    texts = [
                        (f'Think about rounding your wrists!', (10, 540), 0.8),
                        (f'Try to get your wrist angles', (10, 580), 0.8),
                        (f'closer to 16', (10, 610), 0.8)
                    ]
                    rectangles(texts)
                elif (120 < leftElbowAngle < 180 and 120 < rightElbowAngle < 180 and 3 < rightWristAngle < 37 and 3 < leftWristAngle < 37):
                    score4 = 70
                    texts = [
                        (f'Think about rounding your elbows!', (10, 540), 0.8),
                        (f'Try to get your elbow angles', (10, 580), 0.8),
                        (f'closer to 150', (10, 610), 0.8)
                    ]
                    rectangles(texts)
                elif (110 < leftElbowAngle < 190 and 110 < rightElbowAngle < 190 and 0 < rightWristAngle < 47 and 0 < leftWristAngle < 47):
                    score4 = 50
                    texts = [
                        (f'Think about rounding your elbows!', (10, 540), 0.8),
                        (f'Try to get your elbow angles', (10, 580), 0.8),
                        (f'closer to 134', (10, 610), 0.8)
                    ]
                    rectangles(texts)

            else:
                score4 = 0
                print("not in fifth position")
                texts = [
                    (f'Please mirror the dancer example', (10, 540), 0.8),
                ]
                rectangles(texts)
            return score4
def scorePasse():
    if check_landmarks_not_empty(leftFootIndex, leftHeel, rightKnee, rightHip, rightHeel, rightFootIndex, leftWrist, nose, rightShoulder, leftShoulder, rightWrist, leftHip):
        if (leftFootIndex[0] > leftHeel[0] and rightKnee[0] < rightHip[0] and rightHeel[1] < leftKnee[1] and rightHeel[0] < rightFootIndex[0]):
            if (leftWrist[1] < nose[1]  and leftWrist[1] < nose[1]): #and rightShoulder[0] < rightWrist[0] < leftShoulder[0] and nose[1] < rightWrist[1] < rightHip[1]):
                if ( 140 < leftElbowAngle < 160 and 140 < rightElbowAngle < 160 and 9 < rightWristAngle < 25 and 9 < leftWristAngle < 25):
                    score5 = 100
                    texts = [
                        (f'Excellent!', (10, 540), 0.8),
                    ]
                    rectangles(texts)
                elif (130 < leftElbowAngle < 170 and 130 < rightElbowAngle < 170 and 5 < rightWristAngle < 27 and 5 < leftWristAngle < 27):
                    score5= 90
                    texts = [
                        (f'Think about rounding your wrists!', (10, 540), 0.8),
                        (f'Try to get your wrist angles', (10, 580), 0.8),
                        (f'closer to 16', (10, 610), 0.8)
                    ]
                    rectangles(texts)
                elif (120 < leftElbowAngle < 180 and 120 < rightElbowAngle < 180 and 3 < rightWristAngle < 37 and 3 < leftWristAngle < 37):
                    score5 = 70
                    texts = [
                        (f'Think about rounding your elbows!', (10, 540), 0.8),
                        (f'Try to get your elbow angles', (10, 580), 0.8),
                        (f'closer to 150', (10, 610), 0.8)
                    ]
                    rectangles(texts)
                elif (110 < leftElbowAngle < 190 and 110 < rightElbowAngle < 190 and 0 < rightWristAngle < 47 and 0 < leftWristAngle < 47):
                    score5 = 50
                    texts = [
                        (f'Think about rounding your elbows!', (10, 540), 0.8),
                        (f'Try to get your elbow angles', (10, 580), 0.8),
                        (f'closer to 134', (10, 610), 0.8)
                    ]
                    rectangles(texts)
            else:
                score5 = 0
                print("idfjbhefuogvh")
                texts = [
                    (f'Please mirror the dancer example', (10, 540), 0.8),
                ]
                rectangles(texts)
            return score5

def totalScore(score1, score2, score3, score4, score5):
    return score1 + score2 + score3 + score4 + score5

def startingScreen():
    print("Nothing")
def start_timer():
    global score1
    global score2
    global score3
    global score4
    global score5
    threading.Timer(0, startingScreen).start()
    threading.Timer(30, scoreFirst).start()
    threading.Timer(60, scoreSecond).start()
    threading.Timer(90, scoreFourth).start()
    threading.Timer(120, scoreFifth).start()
    threading.Timer(150, scorePasse).start()
    threading.Timer(180, totalScore(score1,score2,score3,score4,score5)).start()

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
            texts = [
                ("Welcome to Ballet Bonanza!", (60, 40), 1.2),
                ("Get ready! In 30 seconds you", (94, 530), 1.0),
                ("should be prepared with an open space to", (8, 580), 0.9),
                ("pose", (310, 630), 1.0)
            ]

            for text, (x, y), scale in texts:
                (w, h), b = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)

                top_left = (x, y - h - b)
                bottom_right = (x + w, y + b)
                cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)
                cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
        elif(position_phase == 2):
            dance_position = "First Position"
            image_path = 'firstPosition.png'
            doForAll(frame,remaining_time, dance_position)
            theScore = scoreFirst()
            (tw1, th1), b1 = cv2.getTextSize('Your score is : 000',  cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (10, 500 - th1 - b1), (10 + tw1, 500 + b1), (255, 255, 255), -1)
            cv2.putText(frame, 'Your score is : ' + str(theScore), (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            overlay_image = cv2.imread(image_path)
            overlay_image = cv2.resize(overlay_image, (200, 400))
            cv2.imshow('Example', overlay_image)
        elif(position_phase == 3):
            dance_position = "Second Position"
            image_path = 'secondPosition.png'
            doForAll(frame, remaining_time, dance_position)
            theScore = scoreSecond()
            (tw2, th2), b2 = cv2.getTextSize('Your score is : 000', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (10, 500 - th2 - b2), (10 + tw2, 500 + b2), (255, 255, 255), -1)
            cv2.putText(frame, 'Your score is : ' + str(theScore), (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            overlay_image = cv2.imread(image_path)
            overlay_image = cv2.resize(overlay_image, (200, 400))
            cv2.imshow('Example', overlay_image)
        elif (position_phase == 4):
            dance_position = "Fourth Position"
            image_path = 'fourthPosition.jpg'
            doForAll(frame, remaining_time, dance_position)
            theScore = scoreFourth()
            (tw3, th3), b3 = cv2.getTextSize('Your score is : 000', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (10, 500 - th3 - b3), (10 + tw3, 500 + b3), (255, 255, 255), -1)
            cv2.putText(frame, 'Your score is : ' + str(theScore), (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            overlay_image = cv2.imread(image_path)
            overlay_image = cv2.resize(overlay_image, (200, 400))
            cv2.imshow('Example', overlay_image)
        elif (position_phase == 5):
            dance_position = "Fifth Position"
            image_path = 'fifthPosition.png'
            doForAll(frame, remaining_time, dance_position)
            theScore = scoreFifth()
            (tw, th), b = cv2.getTextSize('Your score is : 000', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (10, 500 - th - b), (10 + tw, 500 + b), (255, 255, 255), -1)
            cv2.putText(frame, 'Your score is : ' + str(theScore), (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            overlay_image = cv2.imread(image_path)
            overlay_image = cv2.resize(overlay_image, (200, 400))
            cv2.imshow('Example', overlay_image)
        elif (position_phase == 6):
            dance_position = "Right Passe"
            image_path = 'rightPasse.png'
            doForAll(frame, remaining_time, dance_position)
            theScore = scorePasse()
            (tw, th), b = cv2.getTextSize('Your score is : 000', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (10, 500 - th - b), (10 + tw, 500 + b), (255, 255, 255), -1)
            cv2.putText(frame, 'Your score is : ' + str(theScore), (10, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            overlay_image = cv2.imread(image_path)
            overlay_image = cv2.resize(overlay_image, (200, 400))
            cv2.imshow('Example', overlay_image)
        elif(position_phase == 7):
            print(totalScore(score1,score2,score3,score4,score5))
            #print("The score should be printed here and also maybe pass in all the previous scores somehow")
            (tw, th), tb = cv2.getTextSize('Your cumulative score is: 000', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (10, 60 - th - b), (10 + tw, 60 + tb), (255, 255, 255), -1)
            cv2.putText(frame, 'Your cumulative score is: '+str(totalScore(score1,score2,score3,score4,score5)), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
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


