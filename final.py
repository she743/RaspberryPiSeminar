# 1. import packages
import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import RPi.GPIO as gpio
import time

# 2. define hand pose determination function
def angle (dict1, dict2, dict3):
  p1 = np.array([dict1.x, dict1.y])
  p2 = np.array([dict2.x, dict2.y])
  p3 = np.array([dict3.x, dict3.y])
  l1 = p1-p2
  l2 = p3-p2
  cosine_angle = np.dot(l1,l2)/(np.linalg.norm(l1)*np.linalg.norm(l2))
  angle = np.arccos(cosine_angle)
  return angle # radian

def handpose (angle1, angle2, angle3): #thumb, index, middle angle each
  state = 0
  if angle3 > 0.87:
    state = 0 # streched, go
  elif angle2 > 0.87:
    state = 1 # index finger, right turn
  elif angle1 > 2:
    state = 2 # thumb, left turn
  else:
    state = 3 #fist, stop
  return state

# 3. gpio setup
gpio.setmode(gpio.BOARD)
gpio.setup(11, gpio.OUT)
gpio.setup(13, gpio.OUT)
gpio.setup(19, gpio.OUT)
gpio.setup(21, gpio.OUT)

pwm1 = gpio.PWM(11,1000)
pwm2 = gpio.PWM(13,1000)
pwm3 = gpio.PWM(19,1000)
pwm4 = gpio.PWM(21,1000)
pwm1.start(0)
pwm2.start(0)
pwm3.start(0)
pwm4.start(0)
# pwm 1, 2 for right, pwm 3, 4 for left // 1,3 forward 2,4 backward

# 4. get frame & handpose using CV & mediapipe
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.3) as
hands:
while cap.isOpened():
  ret, frame = cap.read()
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  results = hands.process(image) # mediapipe processing
  image.flags.writeable =True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_hand_landmarks: #do if hands are detected
    handmarks = results.multi_hand_landmarks
    for handLandmarks in handmarks:
      mp_drawing.draw_landmarks(frame, handLandmarks, mp_hands.HAND_CONNECTIONS) # mark detected points on camera frame
    # value example) handmarks[mp_hands.HandLandmark.WRIST.value].landmark[0].x
    # cf) results.multi_handedness: # returns classification(left or right)-> index 0 or 1, score, label
    # results.multi_handedness[0].classification[0].label
    
    # determining pose of hand
    if len(handmarks[0].landmark) > 20:
      agl1 = angle(handmarks[0].landmark[mp_hands.HandLandmark.WRIST.value],
                   handmarks[0].landmark[mp_hands.HandLandmark.THUMB_MCP.value],
                   handmarks[0].landmark[mp_hands.HandLandmark.THUMB_TIP.value])
      agl2 = angle(handmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP.value],
                   handmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP.value],
                   handmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value])
      agl3 = angle(handmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value],
                   handmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP.value],
                   handmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP.value])
      pose_state = handpose(agl1, agl2, agl3) # get handpose using pre-defined functions
    
    else:
      pose_state = 0

# 5. write pwm
    if pose_state == 0: # streched, go
      pwm1.ChangeDutyCycle(100)
      pwm2.ChangeDutyCycle(0)
      pwm3.ChangeDutyCycle(100)
      pwm4.ChangeDutyCycle(0)
      print ("go")
    elif pose_state == 1: #right turn
      pwm1.ChangeDutyCycle(100)
      pwm2.ChangeDutyCycle(0)
      pwm3.ChangeDutyCycle(0)
      pwm4.ChangeDutyCycle(100)
      print("right turn")
    elif pose_state == 2: #left turn
      pwm1.ChangeDutyCycle(0)
      pwm2.ChangeDutyCycle(100)
      pwm3.ChangeDutyCycle(100)
      pwm4.ChangeDutyCycle(0)
      print("left turn")
    elif pose_state == 3: # stop
      pwm1.ChangeDutyCycle(0)
      pwm2.ChangeDutyCycle(0)
      pwm3.ChangeDutyCycle(0)
      pwm4.ChangeDutyCycle(0)
      print("stop")

  cv2.imshow('Hand Tracking', frame)

# 6. exit, release camera
  if cv2.waitKey(10) & 0xFF == ord('q'):
    gpio.cleanup()
    break

cap.release()
cv2.destroyAllWindows()
