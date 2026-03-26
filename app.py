import streamlit as st
import pickle
import numpy as np

# 🔥 Title (TOP la varanum)
st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer will leave or stay based on input data")

# 🔥 Load model
model = pickle.load(open('model.pkl', 'rb'))

# 🔥 Inputs (clear labels)
tenure = st.number_input("Tenure (months)", min_value=0.0)
monthly = st.number_input("Monthly Charges (₹)", min_value=0.0)
total = st.number_input("Total Charges (₹)", min_value=0.0)

# 🔥 Predict button
if st.button("Predict"):

    # Input data
    input_data = np.array([[float(tenure), float(monthly), float(total)]], dtype=np.float64)

    # Probability + Threshold
    y_prob = model.predict_proba(input_data)[:,1]
    prediction = (y_prob > 0.55).astype(int)

    # 🔥 Output (better style)
    if prediction[0] == 1:
        st.error(f"Customer will churn ❌\nProbability: {y_prob[0]*100:.2f}%")
    else:
        st.success(f"Customer will stay ✅\nProbability: {y_prob[0]*100:.2f}%")

    # 🔥 Risk indicator
    if y_prob[0] > 0.7:
        st.warning("⚠️ High risk customer!")

    # 🔥 Show input summary
    st.markdown("---")
    st.markdown("👨‍💻 Built by Mahesh 🚀")
    st.write("### 🔍 Input Summary")
    st.write(f"Tenure: {tenure}, Monthly: {monthly}, Total: {total}")
