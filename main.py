import cv2
import mediapipe as mp
import numpy as np
from threading import Timer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

def scoreFirst():
    # First Position
    if (rightFootIndex[0] < rightHeel[0] and leftFootIndex[0] > leftHeel[0] and leftIndex[1] > leftShoulder[1] and
            rightIndex[1] > rightShoulder[1]):
        if (135 < leftElbowAngle < 165 and 135 < rightElbowAngle < 165 and 20 < leftWristAngle < 35 and 20 < rightWristAngle < 35):
            #return cv2.putText(frame, 'First Position', (300, 50), 4, 1, (0, 0, 0), 2, cv2.LINE_AA)
            print("YOOOOO")


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


        #Start checks here
        t = Timer(30, scoreFirst)
       # image = cv2.putText(frame, 'Time (30 seconds till score: '+t, (300, 50), 4, 1, (0, 0, 0), 2, cv2.LINE_AA)
        t.start()
        while(t == False):
            image = cv2.putText(frame, 'Please demonstrate First Position', (20, 50), 4, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    # q key is quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Post release the cap object
vid.release()
# vid2.release()


