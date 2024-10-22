import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn
sklearn.set_config(transform_output='pandas')
import requests


st.write("""# Исследование  SVD""")

url = st.sidebar.text_input("Введите ссылку на изображение")
def load_image_from_url(url):
    try:
        image = io.imread(url)
        return image
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return None

image = load_image_from_url(url)

if image is not None:
    st.image(image)

    if len(image.shape) == 2:
        image = image[:, :]
        U, sing_vals, V = np.linalg.svd(image)
        sigma = np.zeros(shape = image.shape)
        np.fill_diagonal(sigma, sing_vals)
    else:
        image = image[:, :, 2]
        U, sing_vals, V = np.linalg.svd(image)
        sigma = np.zeros(shape = image.shape)
        np.fill_diagonal(sigma, sing_vals)


    st.write("### Необходимо задать топ k сингулярных чисел")

    top_k = st.slider("Задайте топ k сингулярных чисел", min_value=10, max_value=image.shape[0], value=image.shape[0]//2)
    trunc_U = U[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_V = V[:top_k, :]

    button_2 = st.button('Вывести новое изображение')
    if button_2:
        plt.figure(figsize=(50, 40))
        plt.imshow(trunc_U@trunc_sigma@trunc_V, cmap='grey')
        st.write("#### Новое изображение после преобразования")
        st.pyplot(plt)


else:
    st.warning("Не удалось загрузить изображение. Проверьте URL.")