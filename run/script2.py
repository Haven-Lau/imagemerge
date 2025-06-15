import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import hashlib
from PIL import Image, ImageGrab
from io import BytesIO

def get_file_hash(file):
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()

st.set_page_config(page_title="🧵 Vertical Screenshot Stitcher", layout="wide")
st.title("🧵 Vertical Screenshot Stitcher")
st.write("Paste screenshots using **Ctrl+V** (via Paste button) or upload multiple images in order (top to bottom).")

# Initialize session state
if "images" not in st.session_state:
    st.session_state.images = []
if "stitched_result" not in st.session_state:
    st.session_state.stitched_result = None
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Paste from clipboard
if st.button("📋 Paste Image from Clipboard"):
    try:
        img = ImageGrab.grabclipboard()
        if isinstance(img, Image.Image):
            st.session_state.images.append(img.convert("RGB"))
            st.success("✅ Image pasted from clipboard.")
        else:
            st.warning("⚠️ No image found in clipboard.")
    except Exception as e:
        st.error(f"Error pasting image: {e}")

# Upload files with deduplication
uploaded_files = st.file_uploader(
    "📂 Upload screenshots (top to bottom)", type=["png", "jpg", "jpeg"],
    accept_multiple_files=True, key=st.session_state.uploader_key
)
if uploaded_files:
    new_files = [
        (file.name, Image.open(file).convert("RGB"))
        for file in uploaded_files
        if file.name not in st.session_state.uploaded_filenames
    ]

    # Sort new files by filename
    new_files.sort(key=lambda x: x[0])

    # Extend session state
    st.session_state.uploaded_filenames.extend([name for name, _ in new_files])
    st.session_state.images.extend([img for _, img in new_files])

# --- VIDEO UPLOAD + CONTROLS ---
st.subheader("🎥 Upload a Video for Frame Extraction")
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")

# Cache the temp file path using session_state
if video_file is not None:
    video_hash = get_file_hash(video_file)

    if "video_hash" not in st.session_state or st.session_state.video_hash != video_hash:
        # Clean up old temp file
        if "video_temp_path" in st.session_state:
            try:
                os.unlink(st.session_state.video_temp_path)
            except Exception as e:
                st.warning(f"Could not delete previous temp video file: {e}")

        # Create new temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.close()

        # Store path and hash
        st.session_state.video_temp_path = tfile.name
        st.session_state.video_hash = video_hash

    video_path = st.session_state.video_temp_path
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("❌ Failed to open video.")
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps

        #st.video(video_path)
        st.markdown(f"**Video Info:** 🧮 {total_frames} frames &nbsp;&nbsp; 🎞 {fps:.2f} FPS &nbsp;&nbsp; ⏱ {duration:.2f} seconds")

        # --- Initialize State ---
        if "start_frame" not in st.session_state:
            st.session_state.start_frame = 0
        if "end_frame" not in st.session_state:
            st.session_state.end_frame = total_frames - 1
        if "selected_time" not in st.session_state:
            st.session_state.selected_time = 0.0

        # --- Timestamp slider + manual time input ---
        st.markdown("### 🕹 Select Timestamp")

        col_slider, col_time = st.columns([3, 1])
        with col_slider:
            selected_time = st.slider(
                "🕹 Select timestamp (seconds)", 0.0, duration,
                float(st.session_state.selected_time), step=0.1,
                label_visibility="collapsed"
            )

        with col_time:
            # Extract current time components
            total_ms = int(selected_time * 1000)
            current_minutes = total_ms // 60000
            current_seconds = (total_ms % 60000) // 1000
            current_milliseconds = total_ms % 1000

            col_min, col_sec, col_ms = st.columns(3)
            minutes_input = col_min.number_input(
                "Min", min_value=0, max_value=int(duration)//60,
                value=current_minutes, step=1, key="min_input"
            )
            seconds_input = col_sec.number_input(
                "Sec", min_value=0, max_value=59,
                value=current_seconds, step=1, key="sec_input"
            )
            ms_input = col_ms.number_input(
                "Ms", min_value=0, max_value=999,
                value=current_milliseconds, step=1, key="ms_input"
            )

        # Compute total time from manual input
        manual_time = minutes_input * 60 + seconds_input + (ms_input / 1000)
        manual_time = min(manual_time, duration)

        # Sync: If manual time and slider time differ, update the session state
        if abs(manual_time - st.session_state.selected_time) > 0.1:
            st.session_state.selected_time = manual_time
            st.rerun()

        selected_time = manual_time
        selected_frame = int(selected_time * fps)

        # --- Show Time Info ---
        minutes = int(selected_time) // 60
        seconds = int(selected_time) % 60
        st.write(f"🧷 Current Frame Index: {selected_frame} | 🕒 {selected_time:.1f}s ({minutes}:{seconds:02d})")

        # --- Show preview frame ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"Preview at Frame {selected_frame}", use_container_width=False)
        else:
            st.warning("⚠️ Failed to extract preview frame.")

        # --- Frame jump controls ---
        st.markdown("### 🔁 Frame Jump")
        jump_amount = st.slider("Jump Forward/Backward by N frames", 1, 100, 10)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏩ Jump Forward"):
                new_time = min(selected_time + (jump_amount / fps), duration)
                st.session_state.selected_time = new_time
                st.rerun()
        with col2:
            if st.button("⏪ Jump Backward"):
                new_time = max(selected_time - (jump_amount / fps), 0.0)
                st.session_state.selected_time = new_time
                st.rerun()

        # --- Set start/end from current frame ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📍 Set Current as Start Frame"):
                st.session_state.start_frame = selected_frame
        with col2:
            if st.button("🎯 Set Current as End Frame"):
                st.session_state.end_frame = selected_frame

        # --- Manual Start/End Frame Inputs ---
        st.markdown("### 🎞 Frame Range Selection")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.start_frame = st.number_input("Start Frame", min_value=0, max_value=total_frames-1,
                                                           value=st.session_state.start_frame, step=1)
        with col2:
            st.session_state.end_frame = st.number_input("End Frame", min_value=0, max_value=total_frames-1,
                                                         value=st.session_state.end_frame, step=1)

        # --- Sampling Slider ---
        sampling_step = st.slider("🧮 Sampling Step (Use every X-th frame)", min_value=1, max_value=50, value=1)
        st.markdown(f"📊 Effective frames selected: **{len(range(st.session_state.start_frame, st.session_state.end_frame + 1, sampling_step))}**")

        if st.button("🎞 Extract Frames from Range"):
            st.info("⏳ Extracting frames...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.start_frame)
            current_frame = st.session_state.start_frame
            frames_to_add = []

            while current_frame <= st.session_state.end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                if (current_frame - st.session_state.start_frame) % sampling_step == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    frames_to_add.append(pil_img)
                current_frame += 1

            cap.release()  # release before unlink
            try:
                os.unlink(video_path)
            except Exception as e:
                st.warning(f"Could not delete temporary video file: {e}")

            st.session_state.images.extend(frames_to_add)
            st.success(f"✅ Added {len(frames_to_add)} frames.")


# Clear all
if st.button("🗑 Clear All Images"):
    st.session_state.images.clear()
    st.session_state.stitched_result = None
    st.session_state.uploaded_filenames.clear()
    st.session_state.uploader_key += 1  # Force reset of file_uploader
    st.success("✅ All images cleared.")

# Display images with delete buttons
if st.session_state.images:
    st.subheader("🖼️ Input Screenshots")
    for i, img in enumerate(st.session_state.images):
        col1, col2 = st.columns([5, 1])
        with col1:
            buf = BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), caption=f"Image {i+1}", use_container_width=False)
        with col2:
            if st.button(f"❌ Delete Image {i+1}", key=f"del_{i}"):
                st.session_state.images.pop(i)
                st.rerun()

# Options
reverse_input = st.checkbox("🔄 Reverse Image Order", value=False)
gradient_overlay = st.checkbox("🎨 Enable Gradient Overlay on Overlap", value=False)
enable_zoom = st.checkbox("🔍 Enable Scale (Zoom) Compensation", value=False)

# Convert PIL to OpenCV format
def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def stitch_images(images, gradient_overlay=False, enable_zoom=False):
    # Convert first image
    stitched = pil_to_cv2(images[0])
    stitched_offset = np.array([0, 0])
    stitched_corners = [np.array([0, 0]), np.array([stitched.shape[1], stitched.shape[0]])]

    for idx, pil_img in enumerate(images[1:]):
        original_next_img = pil_to_cv2(pil_img)

        # First pass to estimate scale
        gray1 = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        gray2_unscaled = cv2.cvtColor(original_next_img, cv2.COLOR_BGR2GRAY)

        # orb = cv2.ORB_create(
        #     nfeatures=10000,       # More keypoints
        #     scaleFactor=1.2,       # Finer pyramid levels
        #     edgeThreshold=5,       # Detect features closer to image edges
        #     patchSize=31,          # Default patch size
        #     fastThreshold=4        # Lower = more sensitive to low contrast
        # )
        orb = cv2.ORB_create(nfeatures=5000)
        kp1_unscaled, des1_unscaled = orb.detectAndCompute(gray1, None)
        kp2_unscaled, des2_unscaled = orb.detectAndCompute(gray2_unscaled, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        matches_unscaled = bf.match(des1_unscaled, des2_unscaled)
        matches_unscaled = sorted(matches_unscaled, key=lambda x: x.distance)

        scales = []
        if enable_zoom and len(matches_unscaled) >= 2:
            pts1 = np.float32([kp1_unscaled[m.queryIdx].pt for m in matches_unscaled[:50]])
            pts2 = np.float32([kp2_unscaled[m.trainIdx].pt for m in matches_unscaled[:50]])

            for i in range(len(pts1)):
                for j in range(i + 1, len(pts1)):
                    d1 = np.linalg.norm(pts1[i] - pts1[j])
                    d2 = np.linalg.norm(pts2[i] - pts2[j])
                    if d2 > 1e-3:
                        scales.append(d1 / d2)

        scale = np.median(scales) if scales else 1.0

        # Scale the image before actual matching
        if enable_zoom and scale != 1.0:
            next_img = cv2.resize(original_next_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            next_img = original_next_img.copy()

        # Re-run keypoint detection and matching on scaled image
        gray2 = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Rotation-consistent matching
        rotation_threshold = 5  # degrees
        rotation_consistent_matches = []

        for m in matches:
            angle1 = kp1[m.queryIdx].angle
            angle2 = kp2[m.trainIdx].angle
            angle_diff = abs(angle1 - angle2) % 360
            angle_diff = min(angle_diff, 360 - angle_diff)  # shortest angle distance

            if angle_diff <= rotation_threshold:
                rotation_consistent_matches.append(m)

        matches = rotation_consistent_matches


        match_debug = cv2.drawMatches(stitched, kp1, next_img, kp2, matches[:30], None, flags=2)
        st.image(cv2.cvtColor(match_debug, cv2.COLOR_BGR2RGB), caption=f"Top 30 Matches for Image {idx+1}", use_container_width=False)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:30]])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:30]])

        dx = int(np.median(pts1[:, 0] - pts2[:, 0]))
        dy = int(np.median(pts1[:, 1] - pts2[:, 1]))

        st.write(f"Estimated offset (dx, dy) = ({dx}, {dy}), scale = {scale:.3f}")

        # New image's placement in global stitched coordinates
        next_offset = stitched_offset + np.array([dx, dy])
        h2, w2 = next_img.shape[:2]

        # Compute new bounding box to fit both images
        new_topleft = np.minimum(stitched_corners[0], next_offset)
        new_bottomright = np.maximum(stitched_corners[1], next_offset + np.array([w2, h2]))
        new_w = new_bottomright[0] - new_topleft[0]
        new_h = new_bottomright[1] - new_topleft[1]
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        # Compute placement offsets in new canvas
        stitched_canvas_offset = stitched_offset - new_topleft
        next_canvas_offset = next_offset - new_topleft

        # Place stitched image
        h1, w1 = stitched.shape[:2]
        x1, y1 = stitched_canvas_offset.astype(int)
        canvas[y1:y1 + h1, x1:x1 + w1] = stitched

        # Place or blend new image
        x2, y2 = next_canvas_offset.astype(int)
        if gradient_overlay:
            # Calculate overlap between stitched and next image on the canvas
            x_overlap_start = max(x1, x2)
            y_overlap_start = max(y1, y2)
            x_overlap_end = min(x1 + w1, x2 + w2)
            y_overlap_end = min(y1 + h1, y2 + h2)

            if x_overlap_start < x_overlap_end and y_overlap_start < y_overlap_end:
                ow = x_overlap_end - x_overlap_start
                oh = y_overlap_end - y_overlap_start

                # Get regions to blend
                stitched_crop = canvas[y_overlap_start:y_overlap_start + oh, x_overlap_start:x_overlap_start + ow].astype(np.float32)
                next_crop = next_img[y_overlap_start - y2:y_overlap_start - y2 + oh,
                                     x_overlap_start - x2:x_overlap_start - x2 + ow].astype(np.float32)

                # Vertical gradient alpha
                # If second image is below the stitched image, blend from stitched (top) to next (bottom)
                if dy >= 0:
                    alpha = np.linspace(1, 0, oh).reshape(-1, 1)
                else:
                    # If second image is above the stitched image, blend from stitched (bottom) to next (top)
                    alpha = np.linspace(0, 1, oh).reshape(-1, 1)

                alpha = np.repeat(alpha, ow, axis=1)
                alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

                blended = (alpha * stitched_crop + (1 - alpha) * next_crop).astype(np.uint8)
                canvas[y_overlap_start:y_overlap_start + oh, x_overlap_start:x_overlap_start + ow] = blended

                # Paste remaining non-overlapping regions
                # Top part (if any)
                top_h = y_overlap_start - y2
                if top_h > 0:
                    canvas[y2:y2 + top_h, x2:x2 + w2] = next_img[:top_h, :]

                # Bottom part
                bottom_start = y_overlap_start - y2 + oh
                if bottom_start < h2:
                    canvas[y2 + bottom_start:y2 + h2, x2:x2 + w2] = next_img[bottom_start:, :]

                # Left and right parts
                left_w = x_overlap_start - x2
                right_start = x_overlap_end - x2

                if left_w > 0:
                    canvas[y2:y2 + h2, x2:x2 + left_w] = next_img[:, :left_w]
                if right_start < w2:
                    canvas[y2:y2 + h2, x2 + right_start:x2 + w2] = next_img[:, right_start:]
            else:
                # No overlap, just paste
                canvas[y2:y2 + h2, x2:x2 + w2] = next_img


        stitched = canvas
        stitched_offset = new_topleft
        stitched_corners = [new_topleft, new_bottomright]

    return stitched




# Stitch button
if st.button("🧵 Stitch Images"):
    if len(st.session_state.images) >= 2:
        input_imgs = list(reversed(st.session_state.images)) if reverse_input else st.session_state.images
        result = stitch_images(input_imgs, gradient_overlay=gradient_overlay, enable_zoom=enable_zoom)
        st.session_state.stitched_result = result
    else:
        st.warning("Need at least two images to stitch.")

# Show result
if st.session_state.stitched_result is not None:
    st.subheader("🧷 Final Stitched Image")
    stitched_rgb = cv2.cvtColor(st.session_state.stitched_result, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(stitched_rgb)

    buf = BytesIO()
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.image(byte_im, caption="Stitched Result", use_container_width=False)

    st.download_button(
        label="💾 Download Stitched Image (PNG)",
        data=byte_im,
        file_name="stitched_result.png",
        mime="image/png",
        key="download_stitched_image"
    )
