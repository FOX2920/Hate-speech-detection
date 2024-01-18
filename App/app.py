import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Tạo sidebar cho upload file
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Chọn một file CSV", type=["csv"])

# Kiểm tra xem đã upload file chưa
if uploaded_file is not None:
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(uploaded_file)

    # Hiển thị dữ liệu
    st.subheader("Dữ liệu từ file CSV")
    st.write(df)

    # Thống kê số lượng nhãn
    st.subheader("Thống kê số lượng nhãn")
    label_counts = df['label_id'].value_counts()
    
    # Hiển thị số lượng free_text của mỗi nhãn
    st.write("Số lượng nhãn OFFENSIVE (2):", label_counts.get(2, 0))
    st.write("Số lượng nhãn Clean (0):", label_counts.get(0, 0))
    st.write("Số lượng nhãn OFFENSIVE (1):", label_counts.get(1, 0))

    # Hiển thị biểu đồ thống kê
    st.subheader("Biểu đồ thống kê")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='label_id', data=df)
    st.pyplot()

else:
    st.warning("Vui lòng upload file CSV.")
