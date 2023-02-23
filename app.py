
import streamlit as st
from pathlib import Path
from keras.preprocessing import image
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings(action='ignore')


import seaborn as sns
sns.set_style('whitegrid')


st.sidebar.title("Phân loại động vật với thuật toán SVM")
st.sidebar.markdown("Hình ảnh của bạn là: ")
st.sidebar.markdown("✅Mèo 🚫Chó 🍄Cừu")

#đọc dữ liệu hình ảnh, chuyển đổi
p = Path("Train/")
dirs = p.glob("*")
labels_dict = {'cat':0,'dog':1,'sheep':2 }

image_data = []
labels = []
for folder_dir in dirs:
    #print(str(folder_dir))
    label = str(folder_dir).split("\\")[-1][:-1]
    for img_path in folder_dir.glob("*"):
        img = image.load_img(img_path, target_size=(32,32))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        labels.append(labels_dict[label])
#đọc dữ liệu cho test
p_test = Path("Test/")
dirs_test = p_test.glob("*")
labels_dict_test = {'cat':0,'dog':1,'sheep':2 }
image_data_test = []
labels_test = []
for folder_dir in dirs_test:
    #print(str(folder_dir))
    label = str(folder_dir).split("\\")[-1][:-1]
    for img_path in folder_dir.glob("*"):
        img = image.load_img(img_path, target_size=(32,32))
        img_array = image.img_to_array(img)
        image_data_test.append(img_array)
        labels_test.append(labels_dict[label])

## Chuyển đổi dữ liệu thành mảng numpy train
image_data = np.array(image_data, dtype='float32')/255.0
labels = np.array(labels)

## Chuyển đổi dữ liệu thành mảng numpy train
image_data_test = np.array(image_data_test, dtype='float32')/255.0
labels_test = np.array(labels_test)
## Chuyển đổi dữ liệu cho phân loại Một vs Một

#train
M = image_data.shape[0]
image_data = image_data.reshape(M,-1)

#test
M1 = image_data_test.shape[0]
image_data_test = image_data_test.reshape(M1,-1)

number_of_classes = len(np.unique(labels))
print(image_data)
print(labels)

class_names = ['Cat', 'Dog','Sheep']
svm_sklearn1 = pickle.load(open("models/svm_sklearn.pkl","rb"))

df_labels = pd.DataFrame(
    labels,
    columns=['label']
)

image = Image.open('imagemodels/cfm_sklearn.png')
image1 = Image.open('imagemodels/Multiclass ROC sklearn.png')

col1, col2 = st.columns(2)
with col1:
    st.header("Phần trăm dữ liệu")

    labels_circe =  'Cats','Dogs','Sheeps'
    sizes = [2000,2000,2000]
    explode = (0, 0.05,0.05)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots(1)
    ax1.pie(sizes, explode=explode, labels=labels_circe, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)
with col2:
    st.header("Số lượng dữ liệu")
    fig2 = plt.figure(figsize=(8, 6))
    sns.countplot(data=df_labels, x='label')
    st.pyplot(fig2)
def plot_metrics(metrics_list):
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if 'Confusion Matrix' in metrics_list:

        st.subheader("Confusion Matrix")
        st.image(image, caption='Ma trận nhầm lẫn với tập test')

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        st.image(image1, caption='Đường cong ROC với tập test')

class_names = ['Not Spam', 'Spam']

st.sidebar.subheader("Choose Classifier")

classifier = st.sidebar.selectbox("Classification Algorithms",
                                     ("Support Vector Machine (thư viện)",
                                         "Support Vector Machine (Tự xây dựng)"
                                      ))

if classifier == 'Support Vector Machine (thư viện)':

    metrics = st.sidebar.multiselect("Chọn chỉ số lập biểu đồ?",
                                     ('Confusion Matrix','ROC Curve'))

    st.subheader("SVM")
    model = svm_sklearn1
    accuracy = model.score(image_data_test, labels_test)
    #y_pred = model.predict(image_data_test)
    st.write("Accuracy ", accuracy.round(3))
    plot_metrics(metrics)





