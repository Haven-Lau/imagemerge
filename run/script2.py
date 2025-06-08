import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageGrab
from io import BytesIO

st.set_page_config(page_title="🧵 Vertical Screenshot Stitcher", layout="wide")
st.title("🧵 Vertical Screenshot Stitcher")
st.write("Paste screenshots using **Ctrl+V** (via Paste button) or upload multiple images in order (top to bottom).")

# Initialize session state
if "images" not in st.session_state:
    st.session_state.images = []
if "stitched_result" not in st.session_state:
    st.session_state.stitched_result = None
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = set()
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
    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_filenames:
            img = Image.open(file).convert("RGB")
            st.session_state.images.append(img)
            st.session_state.uploaded_filenames.add(file.name)

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

        orb = cv2.ORB_create(1000)
        kp1_unscaled, des1_unscaled = orb.detectAndCompute(gray1, None)
        kp2_unscaled, des2_unscaled = orb.detectAndCompute(gray2_unscaled, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
        if gradient_overlay and 0 <= y2 - y1 < h1 and 0 <= x2 - x1 < w1:
            blend_h = min(h1 - (y2 - y1), h2)
            blend_w = min(w1 - (x2 - x1), w2)

            alpha = np.linspace(1, 0, blend_h).reshape(-1, 1)
            alpha = np.repeat(alpha, blend_w, axis=1)
            alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

            blended = (
                alpha * canvas[y2:y2 + blend_h, x2:x2 + blend_w].astype(np.float32) +
                (1 - alpha) * next_img[:blend_h, :blend_w].astype(np.float32)
            ).astype(np.uint8)

            canvas[y2:y2 + blend_h, x2:x2 + blend_w] = blended
            if blend_h < h2:
                canvas[y2 + blend_h:y2 + h2, x2:x2 + blend_w] = next_img[blend_h:, :blend_w]
            if blend_w < w2:
                canvas[y2:y2 + h2, x2 + blend_w:x2 + w2] = next_img[:h2, blend_w:]
        else:
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
