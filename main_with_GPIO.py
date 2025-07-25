import time
import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
import RPi.GPIO as GPIO
import threading

### ---------- 1. 시스템 및 모델 준비 ---------- ###
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()
time.sleep(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
if file.size == 0:
    print("gesture_train.csv 파일을 로드하지 못했습니다.")
    exit(1)

angle_data = file[:, :-1].astype(np.float32)
label_data = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle_data, cv2.ml.ROW_SAMPLE, label_data)

rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}


# -------- GPIO --------

# switch button Pin
rps_button = 23
mjp_button = 22
replay_button = 27
quit_button = 17

# LED Pin
rps_led = 6
mjp_led = 5

# Servo Pin
left_servo_pin = 12
right_servo_pin = 18

# Buzzer Pin
buzzer_pin = 13
rps_sound = [523, 440, 349, 440, 523]

# -------- GPIO set up --------
GPIO.setwarnings(False); GPIO.setmode(GPIO.BCM)

# switch set up
GPIO.setup(rps_button, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(mjp_button, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(replay_button, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(quit_button, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# led set up
GPIO.setup(rps_led, GPIO.OUT)
GPIO.setup(mjp_led, GPIO.OUT)

# servo set up
GPIO.setup(left_servo_pin, GPIO.OUT)
GPIO.setup(right_servo_pin, GPIO.OUT)

left_servo = GPIO.PWM(left_servo_pin, 50)
right_servo = GPIO.PWM(right_servo_pin, 50)

left_servo.start(0)
right_servo.start(0)

# buzzer set up

def play_buzzer(melody):
    GPIO.setup(buzzer_pin, GPIO.OUT)
    buzzer = GPIO.PWM(buzzer_pin, 100)
    buzzer.start(10)
    for fr in melody:
        buzzer.ChangeFrequency(fr)
        time.sleep(0.5)


# ----- 묵찌빠 판정 함수 -----
def compare_rps(a, b):
    if a == b:
        return 'tie'
    win_map = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
    if win_map[a] == b:
        return 'left'
    else:
        return 'right'

def update_attacker(attack, defense):
    # attack: 공격권자 패, defense: 수비자 패
    if attack == defense:
        return 'end'  # 게임 종료(공격권자 승)
    win_map = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
    if win_map[attack] == defense:
        return 'keep'      # 공격권 유지
    else:
        return 'switch'    # 공격권 넘김

### ---------- 2. 상태 및 변수 초기화 ---------- ###
STATE_SELECT = 0
STATE_READY = 1
STATE_PLAY = 2
STATE_RESULT = 3

state = STATE_SELECT
mode = None
attacker = None
round_winner = None
last_result = ''
ready_count = 3
last_time = time.time()
hit = True

### ---------- 3. 메인 루프 ---------- ###
while True:
    frame_rgb = picam2.capture_array()
    frame_rgb = cv2.flip(frame_rgb, 1)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    key = cv2.waitKey(1) & 0xFF

    display = frame_bgr.copy()

    # -- 1) 모드 선택 UI --
    if state == STATE_SELECT:
        cv2.putText(display, "JJANG KKYEON PPO", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 6, cv2.LINE_AA)
        cv2.putText(display, "JJANG KKYEON PPO", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3, cv2.LINE_AA)
        
        cv2.putText(display, "Select Game Mode :", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(display, "Select Game Mode :", (70, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(display, "RPA : Rock-Paper-Scissors", (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(display, "RPA : Rock-Paper-Scissors", (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(display, "MJP : Muk-jji-ppa", (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(display, "MJP : Muk-jji-ppa", (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.putText(display, "Q : Quit", (100, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(display, "Q : Quit", (100, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2, cv2.LINE_AA)
        
        GPIO.output(rps_led, 0)
        GPIO.output(mjp_led, 0)
        
        
        if GPIO.input(rps_button)==GPIO.HIGH:
            mode = 'RPS'
            state = STATE_READY
            ready_count = 3
            last_time = time.time()
            GPIO.output(rps_led, 1)
        elif GPIO.input(mjp_button)==GPIO.HIGH:
            mode = 'MJP'
            state = STATE_READY
            ready_count = 3
            last_time = time.time()
            GPIO.output(mjp_led, 1)
        elif GPIO.input(quit_button)==GPIO.HIGH:
            break

    # -- 2) 게임 준비 카운트다운 --
    elif state == STATE_READY:
        cv2.putText(display, f"Get Ready! {ready_count}", (160, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
        if time.time() - last_time > 1:
            if ready_count == 3:
                threading.Thread(target=play_buzzer, args=(rps_sound,), daemon=True).start()
            ready_count -= 1
            last_time = time.time()
        if ready_count == 0:
            state = STATE_PLAY
            last_time = time.time()
            attacker = None
            round_winner = None
            last_result = ''

    # -- 3) 게임 진행 (가위바위보 / 묵찌빠) --
    elif state == STATE_PLAY:
        results = hands.process(frame_rgb)
        hand_list = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for idx, lm in enumerate(hand_landmarks.landmark):
                    joint[idx] = [lm.x, lm.y, lm.z]
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                angle_arr = np.arccos(np.einsum(
                    'nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]
                ))
                angle_arr = np.degrees(angle_arr)
                
                sample = np.array([angle_arr], dtype=np.float32)
                ret_knn, results_knn, neighbours, dist = knn.findNearest(sample, 3)
                idx_pred = int(results_knn[0][0])
                
                if idx_pred in rps_gesture:
                    hand_list.append({
                        'rps': rps_gesture[idx_pred],
                        'center': (
                            int(hand_landmarks.landmark[0].x * display.shape[1]),
                            int(hand_landmarks.landmark[0].y * display.shape[0])
                        )
                    })
                    cv2.putText(display, rps_gesture[idx_pred].upper(),
                                (hand_list[-1]['center'][0], hand_list[-1]['center'][1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                mp_drawing.draw_landmarks(display, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 두 손 감지 시 게임 판정
        if len(hand_list) >= 2:
            # 왼/오 손 위치 구분
            if hand_list[0]['center'][0] < hand_list[1]['center'][0]:
                left = hand_list[0]['rps']
                right = hand_list[1]['rps']
            else:
                left = hand_list[1]['rps']
                right = hand_list[0]['rps']

            if mode == 'RPS':
                winner = compare_rps(left, right)
                if winner == 'tie':
                    last_result = "Tie!"
                elif winner == 'left':
                    last_result = "Left Wins! Penalty: Right!"
                elif winner == 'right':
                    last_result = "Right Wins! Penalty: Left!"
                round_winner = winner
                state = STATE_RESULT
                last_time = time.time()

            elif mode == 'MJP':
                if attacker is None:
                    winner = compare_rps(left, right)
                    if winner == 'tie':
                        last_result = "Tie! Do again."
                    else:
                        attacker = winner
                        last_result = f"{attacker.title()} Attack!"
                else:
                    if attacker == 'left':
                        attack = left
                        defense = right  
                    else:
                        attack = right
                        defense = left
                    
                    result = update_attacker(attack, defense)
                    
                    if result == 'end':
                        last_result = f"{attacker.title()} Win! Penalty: {'Right' if attacker=='left' else 'Left'}"
                        round_winner = attacker
                        state = STATE_RESULT
                        last_time = time.time()
                    elif result == 'keep':
                        last_result = f"{attacker.title()} Attack (Keep)!"
                    else:
                        attacker = 'right' if attacker == 'left' else 'left'
                        last_result = f"Attack Turn: {attacker.title()}"
        else:
            if mode == 'MJP' and attacker is not None:
                last_result = f"Attack Turn: {attacker.title()}"
            else:
                last_result = "Show both hands!"

        cv2.putText(display, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        cv2.putText(display, last_result, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

    # -- 4) 결과/벌칙, 재시작 UI --
    elif state == STATE_RESULT:
        cv2.putText(display, "GAME OVER", (160, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
        loser = None
        if mode == 'RPS':
            if round_winner == 'tie':
                cv2.putText(display, "Draw!", (220, 210), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 4)
                
            else:
                loser = "Right" if round_winner == 'left' else "Left"
                cv2.putText(display, f"{round_winner.title()} Win!", (180, 210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
                cv2.putText(display, f"Penalty: {loser}!", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                
        elif mode == 'MJP':
            winner = round_winner
            loser = "Right" if winner == 'left' else "Left"
            cv2.putText(display, f"{winner.title()} Win!", (200, 210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 4)
            cv2.putText(display, f"Penalty: {loser}!", (220, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        
        cv2.putText(display, "R: Restart, Q: Quit", (120, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        
        if loser == "Right":
            while hit:
                right_servo.ChangeDutyCycle(7.5) # 90
                time.sleep(0.5)
                right_servo.ChangeDutyCycle(12.5) # 180
                time.sleep(0.5)
                right_servo.ChangeDutyCycle(7.5) # 90
                time.sleep(0.5)
                right_servo.ChangeDutyCycle(0)
                time.sleep(0.5)
                hit = False
                
        elif loser == "Left":
            while hit:
                left_servo.ChangeDutyCycle(7.5) # 90
                time.sleep(0.5)
                left_servo.ChangeDutyCycle(2.5) # 0
                time.sleep(0.5)
                left_servo.ChangeDutyCycle(7.5) # 90
                time.sleep(0.5)
                left_servo.ChangeDutyCycle(0)
                time.sleep(0.5)
                hit = False
        
        if GPIO.input(replay_button)==GPIO.HIGH:
            hit = True
            state = STATE_READY
            ready_count = 3
            last_time = time.time()
            
        elif GPIO.input(quit_button)==GPIO.HIGH:
            hit = True
            time.sleep(0.2)
            state = STATE_SELECT
            mode = None
            attacker = None
            round_winner = None
            last_result = ''
            ready_count = 3
        

    # -- 5) 영상 출력 및 종료 처리 --
    cv2.imshow('Jjang-Kkyeon-Ppo', display)

# 종료
left_servo.stop()
right_servo.stop()
GPIO.cleanup()
hands.close()
picam2.stop()
cv2.destroyAllWindows()
