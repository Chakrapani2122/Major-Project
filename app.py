import cv2
import numpy as np
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, binary_erosion, square
from skimage.measure import label

import streamlit as st

def crop(image, start_y, end_y, start_x, end_x):
    return image[start_y:end_y, start_x:end_x]

def gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def adaptive_histogram_equalization(image):
    return exposure.equalize_adapthist(image, clip_limit=0.03)

def otsu_threshold(image):
    thresh_val = threshold_otsu(image)
    return image > thresh_val

def dilation_erosion(image):
    dilated = binary_dilation(image, square(3))
    return binary_erosion(dilated, square(3))


def labeling(image):
    labeled_img, num_labels = label(image, connectivity=2, return_num=True)
    return labeled_img, num_labels

def main():
    st.title("Object Counting Streamlit App")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Count Objects"):
            cropped_image = crop(image, 100, 380, 380, 600)
            gamma_corrected_img = gamma_correction(cropped_image, gamma=1.2)
            equalized_img = adaptive_histogram_equalization(gamma_corrected_img)
            thresholded_img = otsu_threshold(equalized_img)
            processed_img = dilation_erosion(thresholded_img)
            labeled_img, num_objects = labeling(processed_img)

            st.write(f"Number of objects detected: {num_objects}")

            if st.checkbox("Show Intermediate Images"):
                st.image(gamma_corrected_img, caption="Gamma Corrected", use_column_width=True)
                st.image(equalized_img, caption="Adaptive Histogram Equalization", use_column_width=True)
                st.image(thresholded_img, caption="Otsu's Thresholding", use_column_width=True)
                st.image(processed_img, caption="Dilation and Erosion", use_column_width=True)
                st.image(labeled_img, caption=f"Labeled Objects (Count: {num_objects})", use_column_width=True)

if __name__ == "__main__":
    main()
