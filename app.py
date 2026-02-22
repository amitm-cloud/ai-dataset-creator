import cv2
import numpy as np
import streamlit as st
import tempfile
import os
import subprocess
import streamlit.components.v1 as components

# ================== CONFIG ==================
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DNN_PROTO = "deploy.prototxt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
FFMPEG_PATH = "ffmpeg"

# ================== LOAD ADSENSE SCRIPT ==================
components.html("""
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-XXXXXXXXXXXX"
     crossorigin="anonymous"></script>
""", height=0)

# ================== PAGE LAYOUT ==================
left_col, center_col, right_col = st.columns([1, 3, 1])

# ================== LEFT AD ==================
with left_col:
    components.html("""
    <div style="width:300px; margin:auto;">
        <ins class="adsbygoogle"
             style="display:inline-block;width:300px;height:600px"
             data-ad-client="ca-pub-XXXXXXXXXXXX"
             data-ad-slot="XXXXXXXXXX"></ins>
        <script>
             (adsbygoogle = window.adsbygoogle || []).push({});
        </script>
    </div>
    """, height=650)

# ================== MAIN CONTENT ==================
with center_col:

    st.title("🎥 AI DataSet Creator + Video Frame Reducer + Dual Face Detection")

    # --------- HELPERS ---------
    def mse(img1, img2):
        return ((img1.astype(float) - img2.astype(float)) ** 2).mean()

    def load_haar():
        cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if cascade.empty():
            raise RuntimeError("Failed to load Haar cascade.")
        return cascade

    def load_dnn():
        if not os.path.exists(DNN_PROTO) or not os.path.exists(DNN_MODEL):
            raise RuntimeError("DNN model files not found.")
        return cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)

    def detect_faces(frame_bgr, detector, method="haar", conf_threshold=0.5):
        crops = []
        h, w = frame_bgr.shape[:2]

        if method == "haar":
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            for (x, y, fw, fh) in faces:
                crops.append(frame_bgr[y:y+fh, x:x+fw])
        else:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame_bgr, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
            )
            detector.setInput(blob)
            detections = detector.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crops.append(frame_bgr[y1:y2, x1:x2])

        return crops

    # -------- Unique Face Filter --------
    def is_new_crop(crop, gallery, threshold=0.2):
        if len(gallery) == 0:
            return True

        crop_small = cv2.resize(crop, (64, 64))
        crop_gray = cv2.cvtColor(crop_small, cv2.COLOR_BGR2GRAY)
        crop_gray = cv2.GaussianBlur(crop_gray, (5, 5), 0)

        for g in gallery:
            g_small = cv2.resize(g, (64, 64))
            g_gray = cv2.cvtColor(g_small, cv2.COLOR_BGR2GRAY)
            g_gray = cv2.GaussianBlur(g_gray, (5, 5), 0)

            if mse(crop_gray, g_gray) < threshold * (255 ** 2):
                return False

        return True

    # -------- Collect Faces --------
    def collect_faces_from_video(video_path, detector, method, max_items=30, step=5):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError("Error opening video.")

        gallery = []
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            progress.progress(min(frame_idx / total_frames, 1.0))

            if frame_idx % step != 0:
                continue

            frame_small = cv2.resize(frame, None, fx=0.5, fy=0.5)
            crops = detect_faces(frame_small, detector, method)

            for crop in crops:
                if len(gallery) >= max_items:
                    break
                if is_new_crop(crop, gallery):
                    gallery.append(crop)

            if len(gallery) >= max_items:
                break

        cap.release()
        progress.empty()
        return gallery

    # -------- Filter Similar Frames --------
    def filter_similar_frames(input_path, output_path, similarity_threshold=0.05):
        cap = cv2.VideoCapture(input_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        out.write(prev_frame)

        max_mse = 255 ** 2

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            similarity = mse(prev_gray, gray)

            if similarity > similarity_threshold * max_mse:
                out.write(frame)
                prev_gray = gray

        cap.release()
        out.release()

    # -------- Split Large Video (>200MB) --------
    def split_video_if_large(input_path, max_size_mb=200):

        file_size_mb = os.path.getsize(input_path) / (1024 * 1024)

        if file_size_mb <= max_size_mb:
            return [input_path]

        st.warning(f"Video is {file_size_mb:.2f} MB. Splitting into {max_size_mb}MB parts...")

        output_pattern = input_path.replace(".mp4", "_part_%03d.mp4")

        subprocess.run([
            FFMPEG_PATH,
            "-i", input_path,
            "-c", "copy",
            "-map", "0",
            "-f", "segment",
            "-segment_time", "600",
            "-reset_timestamps", "1",
            output_pattern
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        parts = sorted([
            os.path.join(os.path.dirname(input_path), f)
            for f in os.listdir(os.path.dirname(input_path))
            if "_part_" in f
        ])

        return parts

    # --------- UI ---------
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    threshold = st.slider("Frame similarity threshold", 0.01, 0.10, 0.05, 0.01)

    detector_choice = st.radio(
        "Choose Face Detection Method:",
        ["Haar Cascade (Fast)", "DNN (More Accurate)"]
    )

    if uploaded_file is not None:

        if "Haar" in detector_choice:
            detector = load_haar()
            method = "haar"
        else:
            detector = load_dnn()
            method = "dnn"

        st.video(uploaded_file)

        if st.button("Analyze and Process"):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                tmp_in.write(uploaded_file.read())
                original_path = tmp_in.name

            try:
                video_parts = split_video_if_large(original_path)

                all_original_faces = []
                all_processed_faces = []
                final_outputs = []

                for idx, part_path in enumerate(video_parts):

                    st.info(f"Processing part {idx+1} of {len(video_parts)}...")

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
                        processed_raw_path = tmp_out.name

                    st.info("Detecting faces in original video...")
                    original_faces = collect_faces_from_video(part_path, detector, method)
                    all_original_faces.extend(original_faces)

                    st.info("Compressing video...")
                    filter_similar_frames(part_path, processed_raw_path, threshold)

                    processed_final_path = processed_raw_path.replace(".mp4", "_h264.mp4")

                    subprocess.run([
                        FFMPEG_PATH,
                        "-y",
                        "-i", processed_raw_path,
                        "-vcodec", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        processed_final_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    final_outputs.append(processed_final_path)

                    processed_faces = collect_faces_from_video(processed_final_path, detector, method)
                    all_processed_faces.extend(processed_faces)

                st.success("Processing Complete!")

                for video in final_outputs:
                    st.video(video)

                def show_gallery(title, faces):
                    st.subheader(title)
                    if len(faces) == 0:
                        st.info("No faces detected.")
                    else:
                        cols = st.columns(4)
                        for idx, crop in enumerate(faces):
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            cols[idx % 4].image(crop_rgb, width=150)

                show_gallery("Faces in Original Video", all_original_faces)
                show_gallery("Faces in Processed Video", all_processed_faces)

                st.subheader("Download Processed Video Parts")

                for idx, video_path in enumerate(final_outputs):
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()

                    st.download_button(
                        label=f"Download Processed Video - Part {idx+1}",
                        data=video_bytes,
                        file_name=f"processed_video_part_{idx+1}.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"Error: {e}")

            finally:
                try:
                    os.remove(original_path)
                except:
                    pass

# ================== RIGHT AD ==================
with right_col:
    components.html("""
    <div style="width:300px; margin:auto;">
        <ins class="adsbygoogle"
             style="display:inline-block;width:300px;height:600px"
             data-ad-client="ca-pub-XXXXXXXXXXXX"
             data-ad-slot="XXXXXXXXXX"></ins>
        <script>
             (adsbygoogle = window.adsbygoogle || []).push({});
        </script>
    </div>

    """, height=650)
