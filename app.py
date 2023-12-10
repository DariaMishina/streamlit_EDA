import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)



def process_main_page():
    show_main_page()
    viz_eda()


def show_main_page():
    image = Image.open('UPLIFT.jpg')

    st.set_page_config(
        # layout="wide",
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
    for column in data[['AGE', 'SOCSTATUS_WORK_FL',
       'SOCSTATUS_PENS_FL', 'GENDER', 'CHILD_TOTAL', 'DEPENDANTS',
       'PERSONAL_INCOME', 'LOAN_NUM_CLOSED', 'LOAN_NUM_TOTAL']]:
        st.write(f"## {column} Distribution")
        if data[column].dtype == 'float64' or data[column].dtype == 'int64':
            plt.figure(figsize=(8, 6))
            sns.histplot(data[column], kde=True)
            st.pyplot()
    st.text("в целом, глядя на графики распределений признаков, можно отметить значительное разнообразие значений только "
            "у признаков AGE и PERSONAL_INCOME, все остальные признаки, несмотря на числовой формат, являются категориальными")

    # Шаг 3: Построение матрицы корреляций
    st.subheader('Матрица корреляций')
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()
    st.text("исходя из матрицы корреляций, можно сделать выводы о том, что с целевой переменной признаки скоррелированы слабо, "
            "зато есть сильные корреляции (как прямые, так и обратные) между признаками: 'LOAN_NUM_CLOSED' и 'LOAN_NUM_TOTAL', "
            "'CHILD_TOTAL' и 'DEPENDANTS', 'AGE' и 'SOCSTATUS_WORK_FL', 'AGE' и 'SOCSTATUS_PENS_FL', SOCSTATUS_PENS_FL и 'SOCSTATUS_WORK_FL', "
            "что во всех случаях соответствует и здравому смыслу")

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
    st.text("Несмотря на некоторую некрасивость визуализации зависимости некоторых признаков и таргета, можно сделать интересные выводы о том, "
            "что  отклик в большинстве случаев был зарегистрирован у работающих людей и не пенсионеров, в то время как отклика не было у не работающих и у пенсионеров."
            "Также можно сделать вывод, что откликов больше у более молодых людей, по гендеру значительных различий в отлике не наблюдается, "
            "как и по количеству детей и иждивенцев (но здесь есть выбросы), "
            "также отклик наблюдается у людей с бОльшим доходом (но также присутствует много выбросов. "
            "Интересная картина наблюдается с займами: по количеству закрытых займов значительных различий в отлике не наблюдается, "
            "а вот по общему количеству оклика нет у тех, у кого займов больше (что в целом логично, они и так уже закредитованы)")


if __name__ == "__main__":
    process_main_page()
