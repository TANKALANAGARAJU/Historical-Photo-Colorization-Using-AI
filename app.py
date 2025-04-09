import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os
import io


def is_grayscale(img):
    if len(img.shape) == 3 and img.shape[2] == 3:  # Ensure it's a color image
        if np.all(img[:, :, 0] == img[:, :, 1]) and np.all(img[:, :, 1] == img[:, :, 2]):
            return True  # Image is grayscale
    return False  # Image is color

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

def colorizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    script_dir = os.path.dirname(__file__)
    prototxt = os.path.join(script_dir, "models/models_colorization_deploy_v2.prototxt")
    model = os.path.join(script_dir, "models/colorization_release_v2.caffemodel")
    points = os.path.join(script_dir, "models/pts_in_hull.npy")
    
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    return colorized

st.title("Colorize your Black and White Image")
st.write("This is an app to colorize your B&W images.")

input_images_dir = "Input_images"
input_images = [f for f in os.listdir(input_images_dir) if f.endswith((".jpg", ".png"))]
selected_image = st.sidebar.selectbox("Choose a sample image", ["None"] + input_images)

file = st.sidebar.file_uploader("Or upload an image file", type=["jpg", "png"])

if file is None:
    if selected_image != "None":
        image_path = os.path.join(input_images_dir, selected_image)
        image = Image.open(image_path)
        img = np.array(image)
    else:
        image = None
else:
    image = Image.open(file)
    img = np.array(image)
    
if image:
    st.text("Your original image")
    st.image(image, use_column_width=True)
    
    if not is_grayscale(img):
        st.warning("⚠️ Your image is already colorized!")
    else:
        st.text("Your colorized image")
        color = colorizer(img)
        st.image(color, use_column_width=True)
        
        # Convert colorized image to a downloadable format
        color_pil = Image.fromarray(color)
        buf = io.BytesIO()
        color_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="Download Colorized Image",
            data=byte_im,
            file_name="colorized_image.png",
            mime="image/png"
        )
