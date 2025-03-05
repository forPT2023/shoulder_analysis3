import streamlit as st
import os
import analysis
import visualization

SAVE_DIR = "saved_results"
os.makedirs(SAVE_DIR, exist_ok=True)

st.title("肩関節動作分析アプリ")

# 解析モードの選択
mode = st.radio("解析する動作を選択:", ["肩関節外転", "肩関節屈曲"])

# 左右の選択
side = st.radio("対象の上肢:", ["右", "左"])

uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "avi", "mov"])

if uploaded_file:
    video_path = os.path.join(SAVE_DIR, "uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.write("動画を解析中...")
    output_video_path, angles_data, max_rom = analysis.process_video(video_path, mode, side)

    st.write(f"最大可動域（ROM）: {max_rom:.2f}°")

    fig = visualization.plot_joint_angles(angles_data)
    st.pyplot(fig)

    st.write("📹 関節マーカー付きの解析動画")
    st.video(output_video_path)

