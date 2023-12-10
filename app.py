import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image



def process_main_page():
    show_main_page()
    viz_eda()


def show_main_page():
    image = Image.open('UPLIFT.jpg')

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="EDA for Uplift",
        page_icon=image,

    )

    st.write(
        """
        # Exploratory data analysis: исследуем наши данные, предварительно очищенные и обработанные.
        """
    )

    st.image(image)


@st.cache
# Шаг 1: Загрузка данных
def load_data():
    data = pd.read_csv('total.csv')
    return data


def viz_eda():
    # Шаг 2: Построение графиков распределений признаков
    st.subheader('Графики распределений признаков')
    data = load_data()
    for column in data.columns:
        st.write(f"## {column} Distribution")
        if data[column].dtype == 'float64' or data[column].dtype == 'int64':
            plt.figure(figsize=(8, 6))
            sns.histplot(data[column], kde=True)
            st.pyplot()

    # Шаг 3: Построение матрицы корреляций
    st.subheader('Матрица корреляций')
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()

    # Шаг 4: Построение графиков зависимостей целевой переменной и признаков
    st.subheader('Графики зависимостей целевой переменной и признаков')
    for column in data.columns:
        st.write(f"## {column} vs TARGET")
        if data[column].dtype == 'float64' or data[column].dtype == 'int64':
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='TARGET', y=column, data=data)
            st.pyplot()

    # Шаг 5: Вычисление числовых характеристик
    st.subheader('Числовые характеристики распределения числовых столбцов')
    stats = data.describe()
    st.write(stats)


if __name__ == "__main__":
    process_main_page()
