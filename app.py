import streamlit as st
import joblib

# Load the trained models and TF-IDF vectorizer
@st.cache_resource
def load_models():
    svm_model = joblib.load('svm_model.pkl')
    nb_model = joblib.load('nb_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return svm_model, nb_model, vectorizer

svm_model, nb_model, vectorizer = load_models()

# Custom CSS for a modern design
st.markdown(
    """
    <style>
        /* General Styling */
        .stApp {
            background: linear-gradient(135deg, #f0f4f8 0%, #e0e7ff 100%);
            font-family: 'Inter', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 40px !important; /* Kept as per your code */
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 10px;
            line-height: 1.1;
        }
        .dataset-info {
            text-align: center;
            font-size: 16px;
            color: #475569;
            background: #ffffff;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 20px;
        }
        .sentiment-positive {
            color: #16a34a;
            font-size: 30px !important; /* Increased size */
            font-weight: 600;
            text-align: center;
            animation: fadeIn 0.5s ease-in;
        }
        .sentiment-negative {
            color: #dc2626;
            font-size: 30px !important; /* Increased size */
            font-weight: 600;
            text-align: center;
            animation: fadeIn 0.5s ease-in;
        }

        /* Input and Button Styling */
        .stTextArea textarea {
            background: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 15px;
            font-size: 16px;
            color: #1e293b;
            transition: border-color 0.3s ease;
        }
        .stTextArea textarea:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }
        .stSelectbox > div > div {
            background: #ffffff;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 16px;
            color: #1e293b;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 30px;
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }

        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Responsive Design */
        @media (max-width: 600px) {
            .main-title {
                font-size: 50px !important; /* Kept as per your code */
            }
            .sentiment-positive, .sentiment-negative {
                font-size: 32px !important; /* Adjusted for mobile */
            }
            .stButton>button {
                width: 100%;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<p class="main-title">Nepali Sentiment Analysis</p>', unsafe_allow_html=True)

# Dataset Info
st.markdown(
    """
    <p class="dataset-info">
        ✅ Positive Sentences: <b>15,880</b>     ❌ Negative Sentences: <b>14,408</b>
    </p>
    """,
    unsafe_allow_html=True
)

# Model Selection
model_option = st.selectbox("Choose a Model:", ["SVM", "Naive Bayes"], help="Select the machine learning model for prediction")

# Input Box
user_input = st.text_area(
    "✍️ Enter Nepali Text",
    value="यो फिल्म धेरै लामो र अल्छी लाग्दो छ",
    height=120,
    placeholder="Type your Nepali text here..."
)

# Predict Button
if st.button("🔍 Predict Sentiment"):
    # Transform input
    processed_input = vectorizer.transform([user_input])

    # Make a prediction
    prediction = svm_model.predict(processed_input)[0] if model_option == "SVM" else nb_model.predict(processed_input)[0]

    # Sentiment mapping with emoji
    sentiment_map = {
        -1: ("Negative 😞", "sentiment-negative"),
        0: ("Negative 😞", "sentiment-negative"),
        1: ("Positive 😊", "sentiment-positive")
    }
    sentiment_text, sentiment_class = sentiment_map.get(prediction, ("Unknown Sentiment 🤔", ""))

    # Display sentiment
    st.markdown(f'<p class="{sentiment_class}">Predicted Sentiment: {sentiment_text}</p>', unsafe_allow_html=True)