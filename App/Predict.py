import streamlit as st
from pyspark.ml import PipelineModel

# Load the Spark model
spark_model = PipelineModel.load("path/to/your/spark_model")

# Streamlit app
st.title("Text Classification App")

# Input text box
text_input = st.text_area("Enter text for classification:", "")

# Make predictions
if st.button("Classify"):
    if text_input:
        prediction = spark_model.transform([text_input]).select("prediction").collect()[0][0]
        if prediction == 0.0:
            result = "CLEAN"
        elif prediction == 1.0:
            result = "OFFENSIVE"
        else:
            result = "HATE"

        st.success(f"The text is classified as: {result}")
    else:
        st.warning("Please enter some text for classification.")

