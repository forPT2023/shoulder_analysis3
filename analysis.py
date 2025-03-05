import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess  # メタデータ取得用

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # 関節マーカー描画用

def get_video_rotation(video_path):
    """動画の回転情報を取得する（FFmpeg を使用）"""
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream_tags=rotate", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        rotation = subprocess.check_output(cmd).decode("utf-8").strip()
        return int(rotation) if rotation else 0
    except Exception as e:
        return 0  # エラー時は回転なしと判断

def process_video(video_path, mode, side):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose()
    angles = []
    output_video_path = "saved_results/processed_video.mp4"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # **動画の回転情報を取得**
    rotation = get_video_rotation(video_path)
    
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # **メタデータに基づいて必要な場合のみ回転補正**
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 時計回りに90°
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)  # 180°回転
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 反時計回りに90°

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # --- 必要な関節データ取得 ---
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER if side == "右" else mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW if side == "右" else mp_pose.PoseLandmark.LEFT_ELBOW]
            hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP if side == "右" else mp_pose.PoseLandmark.LEFT_HIP]

            h, w, _ = frame.shape
            def get_pixel_coords(landmark):
                return int(landmark.x * w), int(landmark.y * h)

            # --- 基本軸の計算 ---
            if mode == "肩関節外転":
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

                shoulder_mid = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
                hip_mid = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])

                torso_vector = shoulder_mid - hip_mid
                arm_vector = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])

            else:
                torso_vector = np.array([shoulder.x - hip.x, shoulder.z - hip.z])
                arm_vector = np.array([shoulder.x - elbow.x, shoulder.z - elbow.z])
                shoulder_mid, hip_mid = shoulder, hip

            # --- 角度の計算 ---
            angle = np.degrees(np.arccos(np.dot(arm_vector, torso_vector) /
                        (np.linalg.norm(arm_vector) * np.linalg.norm(torso_vector))))
            
            angles.append(angle)

            # --- 関節マーカーと線の描画 ---
            cv2.circle(frame, get_pixel_coords(shoulder), 5, (0, 0, 255), -1)  # 肩（赤）
            cv2.circle(frame, get_pixel_coords(elbow), 5, (255, 0, 0), -1)  # 肘（青）

            if mode == "肩関節屈曲":
                cv2.circle(frame, get_pixel_coords(hip), 5, (0, 255, 0), -1)  # **股関節（緑）**
                cv2.line(frame, get_pixel_coords(hip_mid), get_pixel_coords(shoulder_mid), (0, 255, 0), 2)  # **屈曲の基本軸（緑）**

            cv2.line(frame, get_pixel_coords(shoulder), get_pixel_coords(elbow), (0, 255, 255), 2)  # 肩-肘（黄色）

            if mode == "肩関節外転":
                cv2.line(frame, (int(hip_mid[0] * w), int(hip_mid[1] * h)), 
                         (int(shoulder_mid[0] * w), int(shoulder_mid[1] * h)), (255, 0, 0), 2)  # **外転の基本軸（青）**

        out.write(frame)

    cap.release()
    out.release()

    return output_video_path, angles, max(angles) if angles else 0

