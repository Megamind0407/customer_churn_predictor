# Customer Churn Prediction System
This is a **Customer Churn Prediction System** built with **Streamlit**, **Machine Learning**, and **Sentiment Analysis**. The system allows users to predict customer churn based on their subscription data and provides personalized intervention recommendations. It also includes an optional sentiment analysis of customer feedback.

## 🚀 Features

- **Churn Prediction**: Predict the likelihood of a customer churning based on features like subscription type, tenure, and services used.
- **Risk Score Calculation**: Calculate a risk score based on the churn probability and customer feedback sentiment.
- **Intervention Recommendations**: Suggest interventions (e.g., discounts, technical support) based on the customer's risk profile.
- **Customer Feedback Analysis**: Perform sentiment analysis on customer feedback to adjust the risk profile.
- **Data Upload**: Upload customer data in CSV format to make predictions for specific customers.
- **Interactive Dashboard**: A user-friendly interface built with Streamlit to allow easy input and data visualization.

## 📊 Dataset
- [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/c/twitter-sentiment-analysis)  
- [IBM Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn) 

## 📷 Screenshots
![image](https://github.com/user-attachments/assets/76177f9e-239f-4929-9815-770725a0eb4d)
![image](https://github.com/user-attachments/assets/1b498b07-7938-494f-afde-5efc44e3ca5c)
![image](https://github.com/user-attachments/assets/dc229c03-876c-41e4-b0f2-3daca8af86f8)


## 🛠️ Tech Stack

- **Frontend**: 
  - [Streamlit](https://streamlit.io/): A Python library to create interactive web applications for machine learning and data science projects.
  
- **Backend**:
  - **Python**: Programming language used for the entire application.
  - **Scikit-learn**: For building and training the churn prediction machine learning model.
  - **Pandas**: For data manipulation and analysis.
  - **NumPy**: For numerical operations.
  
- **Machine Learning**:
  - **Logistic Regression / Random Forest**: Machine learning algorithms for churn prediction.
  
- **Natural Language Processing**:
  - **Transformers** (Hugging Face): For sentiment analysis on customer feedback using pre-trained models like `distilbert-base-uncased-finetuned-sst-2-english`.
  
- **Data Visualization**:
  - **Matplotlib** & **Seaborn**: For plotting and visualizing data trends.

## 🧠 Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Transformers (Hugging Face)
- Matplotlib
- Seaborn


### Clone the repository:
```bash
git clone https://github.com/Megamind0407/customer_churn_predictor.git
cd customer-churn-prediction
```
### Install the required dependencies:
```bash
pip install -r requirements.txt
```
### Run the Streamlit app:
```bash
streamlit run app.py
```
Access the application on your browser at `http://localhost:8501`

## 🧩 Future Enhancements

- Incorporate LLM (e.g., GPT) for smarter sentiment classification  
- Add multilingual tweet handling  
- Deploy as a REST API for third-party integration  

## 📄 License

This project is licensed under the MIT License.

---
