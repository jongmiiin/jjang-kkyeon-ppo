import time
import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp

# ----- 1) Picamera2 설정 -----
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    # main={"format": "RGB888", "size": (1080, 720)}  # 해상도 1080×720
    main={"format": "RGB888", "size": (640, 480)}  # 해상도 640×480
    # main={"format": "RGB888", "size": (320, 240)}  # 해상도 320×240로 낮춰서 속도 최적화
)
picam2.configure(config)
picam2.start()
time.sleep(1)  # Exposure 안정화

# ----- 2) MediaPipe Hands 초기화 -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ----- 3) KNN 학습 데이터 로드 (가위·바위·보) -----
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
if file.size == 0:
    print("gesture_train.csv 파일을 로드하지 못했습니다.")
    exit(1)

angle_data = file[:, :-1].astype(np.float32)
label_data = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle_data, cv2.ml.ROW_SAMPLE, label_data)

# 가위·바위·보 인덱스 매핑
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}


# ----- 4) 메인 루프 -----
while True:
    # 4-1) Picamera2에서 프레임 획득 (RGB)
    frame_rgb = picam2.capture_array()  # shape=(240,320,3), RGB 순서

    # 4-2) MediaPipe Hands로 손 검출
    results = hands.process(frame_rgb)

    # 4-3) OpenCV 용 BGR 변환
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        rps_list = []  # 한 프레임내 여러 손의 가위바위보 결과 저장
        for hand_landmarks in results.multi_hand_landmarks:
            # 4-4) 각 관절 위치 추출
            joint = np.zeros((21, 3))
            for idx, lm in enumerate(hand_landmarks.landmark):
                joint[idx] = [lm.x, lm.y, lm.z]

            # 4-5) 관절 간 벡터와 각도 계산
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle_arr = np.arccos(np.einsum(
                'nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :]
            ))
            angle_arr = np.degrees(angle_arr)  # 라디안을 도 단위로 변경

            # 4-6) KNN으로 추론
            sample = np.array([angle_arr], dtype=np.float32)
            ret_knn, results_knn, neighbours, dist = knn.findNearest(sample, 3)
            idx_pred = int(results_knn[0][0])

            # 4-7) rps_gesture 사전에 해당 인덱스가 있으면, 화면에 텍스트 표시
            if idx_pred in rps_gesture:
                label_text = rps_gesture[idx_pred].upper()
                # 첫 번째 랜드마크(엄지 손목 부근) 위치에 텍스트 표시
                org_x = int(hand_landmarks.landmark[0].x * frame_bgr.shape[1])
                org_y = int(hand_landmarks.landmark[0].y * frame_bgr.shape[0]) + 20
                cv2.putText(
                    frame_bgr,
                    text=label_text,
                    org=(org_x, org_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2
                )
                rps_list.append({'rps': rps_gesture[idx_pred], 'org': (org_x, org_y)})

            # 4-8) 손 랜드마크 선 긋기
            mp_drawing.draw_landmarks(
                frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

        # 4-9) 두 손 이상 검출된 경우 승패 판정 (선택 사항)
        if len(rps_list) >= 2:
            winner = None
            text = ''
            r0 = rps_list[0]['rps']
            r1 = rps_list[1]['rps']
            if r0 == 'rock':
                if r1 == 'rock':
                    text = 'Tie'
                elif r1 == 'paper':
                    text = 'Paper wins'; winner = 1
                else:  # r1 == 'scissors'
                    text = 'Rock wins'; winner = 0
            elif r0 == 'paper':
                if r1 == 'rock':
                    text = 'Paper wins'; winner = 0
                elif r1 == 'paper':
                    text = 'Tie'
                else:  # r1 == 'scissors'
                    text = 'Scissors wins'; winner = 1
            else:  # r0 == 'scissors'
                if r1 == 'rock':
                    text = 'Rock wins'; winner = 1
                elif r1 == 'paper':
                    text = 'Scissors wins'; winner = 0
                else:
                    text = 'Tie'

            # 승자 위치에 “Winner” 표시
            if winner is not None:
                wx, wy = rps_list[winner]['org']
                cv2.putText(
                    frame_bgr,
                    text='Winner',
                    org=(wx, wy + 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(0, 255, 0),
                    thickness=3
                )
            # 화면 중앙 상단에 결과 텍스트 표시
            cv2.putText(
                frame_bgr,
                text=text,
                org=(int(frame_bgr.shape[1]/2) - 100, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=(0, 0, 255),
                thickness=3
            )

    # 5) 결과 영상 표시
    cv2.imshow('RPS Game (Picamera2 + MediaPipe)', frame_bgr)

    # 6) 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----- 7) 자원 해제 -----
hands.close()
picam2.stop()
cv2.destroyAllWindows()
