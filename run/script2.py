import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageGrab
from io import BytesIO

st.set_page_config(page_title="🧵 Vertical Screenshot Stitcher", layout="wide")
st.title("🧵 Vertical Screenshot Stitcher")
st.write("Paste screenshots using **Ctrl+V** (via Paste button) or upload multiple images in order (top to bottom).")

if "images" not in st.session_state:
    st.session_state.images = []

if "stitched_result" not in st.session_state:
    st.session_state.stitched_result = None

# Paste
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

# Upload
uploaded_files = st.file_uploader(
    "📂 Upload screenshots (top to bottom)", type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)
if uploaded_files:
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        st.session_state.images.append(img)

# Clear
if st.button("🗑 Clear All Images"):
    st.session_state.images.clear()
    st.session_state.stitched_result = None
    st.success("✅ All images cleared.")

# Display uploaded images with delete buttons
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
                st.experimental_rerun()


# Options
reverse_input = st.checkbox("🔄 Reverse Image Order", value=False)
gradient_overlay = st.checkbox("🎨 Enable Gradient Overlay on Overlap", value=False)

# PIL → CV2
def pil_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Stitcher
def stitch_images(images, gradient_overlay=False):
    stitched = pil_to_cv2(images[0])
    for idx, pil_img in enumerate(images[1:]):
        next_img = pil_to_cv2(pil_img)
        gray1 = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        match_debug = cv2.drawMatches(stitched, kp1, next_img, kp2, matches[:30], None, flags=2)
        st.image(cv2.cvtColor(match_debug, cv2.COLOR_BGR2RGB), caption=f"Top 30 Matches for Image {idx+1}", use_container_width=False)

        dy_values = [kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1] for m in matches[:30]]
        dy_median = int(np.median(dy_values))
        st.write(f"Estimated vertical offset (dy) = {dy_median}")

        h1, w1 = stitched.shape[:2]
        h2, w2 = next_img.shape[:2]
        new_h = max(h1, dy_median + h2)
        new_w = max(w1, w2)
        output = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        if gradient_overlay and dy_median < h1:
            overlap = h1 - dy_median
            blend_height = min(overlap, h2)
            alpha = np.linspace(1, 0, blend_height).reshape(-1, 1)
            alpha = np.repeat(alpha, w2, axis=1)
            alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

            blended = (
                alpha * stitched[dy_median:dy_median + blend_height, :w2].astype(np.float32) +
                (1 - alpha) * next_img[:blend_height, :w2].astype(np.float32)
            ).astype(np.uint8)

            output[:h1, :w1] = stitched
            output[dy_median:dy_median + blend_height, :w2] = blended
            if blend_height < h2:
                output[dy_median + blend_height:dy_median + h2, :w2] = next_img[blend_height:, :w2]
        else:
            output[:h1, :w1] = stitched
            output[dy_median:dy_median + h2, :w2] = next_img

        stitched = output

    return stitched

# Stitch button
if st.button("🧵 Stitch Images"):
    if len(st.session_state.images) >= 2:
        input_imgs = list(reversed(st.session_state.images)) if reverse_input else st.session_state.images
        result = stitch_images(input_imgs, gradient_overlay=gradient_overlay)
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

