import cv2
import mediapipe as mp
import numpy as np
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

rightWrist = 0
leftWrist = 0
leftIndex = 0
leftPinky = 0
leftAnkle = 0
rightAnkle = 0
red_dot = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)  # draw red dots
green_line = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)

while True:

    # Capture the video frame by frame
    ret, frame = vid.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame)  # result
    image = cv2.putText(frame, 'Your pose is: ', (50, 50), 4, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # red landmarks, and green connection lines
    height, width, _ = frame.shape
    if (results.pose_landmarks):
        # wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * height
        #  nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * height

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

        leftWristAngle = calculate_angle(leftIndex, leftWrist, leftPinky)
        rightWristAngle = calculate_angle(rightIndex, rightWrist, rightPinky)
        leftElbowAngle = calculate_angle(leftWrist, leftElbow, leftShoulder)
        rightElbowAngle = calculate_angle(rightWrist, rightElbow, rightShoulder)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # fix color

        # multiply the position of the text so that it is next to left wrist times webcam dimensions
        cv2.putText(frame, str(leftWristAngle),
                    tuple(np.multiply(leftWrist, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

        cv2.putText(frame, str(rightWristAngle),
                    tuple(np.multiply(rightWrist, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )
        cv2.putText(frame, str(leftElbowAngle),
                    tuple(np.multiply(leftElbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )
        cv2.putText(frame, str(rightElbowAngle),
                    tuple(np.multiply(rightElbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=red_dot,
                               connection_drawing_spec=green_line)

    cv2.imshow('frame', frame)

    # q key is quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Post release the cap object
vid.release()
# vid2.release()


