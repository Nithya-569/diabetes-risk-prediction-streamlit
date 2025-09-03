import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
st.title("Diabetes Prediction using Logistic Regression")
with st.expander("Data Set"):
    df = pd.read_csv(r"C:\Users\msnit\Downloads\diabetes.csv")
    st.dataframe(df)
with st.expander("Generating random values in the DataFrame"):
    st.write(df.sample(10))
with st.expander("Diving deeper into the data"):
    st.write("Description of the data:")
    st.write(df.describe())
    st.write("Checking for null values:")
    st.write(df.isnull().sum())
    st.write("Correlation matrix:")
    st.write(df.corr())
with st.expander("Visualizations"):
    st.write("Scatter plot: Glucose vs BMI colored by Outcome")
    st.scatter_chart(df, x='Glucose', y='BMI', color='Outcome')
col1 = st.text_input("Do you want to see Logistic Regression working? (yes/no)", "yes/no")
if col1.lower() == "yes":
    st.balloons()
    st.subheader("Logistic Regression Model")
    X = df.drop(columns='Outcome', axis=1)
    Y = df['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    with st.expander("Model Performance"):
        st.write("Accuracy:", accuracy_score(Y_test, pred))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(Y_test, pred))
        st.write("Classification Report:")
        st.text(classification_report(Y_test, pred))
    st.subheader("Test with your own data")
    Pregnancies = st.slider("Number of Pregnancies", 0, 20, 1)
    Glucose = st.slider("Glucose Level", 0, 200, 100)
    BloodPressure = st.slider("Blood Pressure", 0, 150, 70)
    SkinThickness = st.slider("Skin Thickness", 0, 100, 20)
    Insulin = st.slider("Insulin Level", 0, 900, 85)
    BMI = st.slider("BMI", 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    Age = st.slider("Age", 0, 120, 30)
    user_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]

    if st.button("Predict"):
        user_pred = model.predict(user_data)
        if user_pred[0] == 0:
            st.success("The person is NOT diabetic")
        else:
            st.error("The person is diabetic")
