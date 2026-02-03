import streamlit as st
import pickle
import base64


def get_base64_bg(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

bg_image = get_base64_bg("background.jpg")

st.set_page_config(
    page_title="Movie Genre Predictor",
    page_icon="🎬",
    layout="centered"
)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    h1, h2, h3 {{
        color: #ffff00;
        text-align: center;
    }}

    .stTextArea textarea {{
        background-color: rgba(2, 6, 23, 0.85);
        color: #e5e7eb;
        border-radius: 10px;
    }}

    .stButton button {{
        background-color: #facc15;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
        height: 3em;
    }}

    .result-box {{
        background-color: rgba(2, 6, 25, 0.9);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        text-align: center;
        font-size: 22px;
    }}
        .intro-text {{
        color: #ffff00;   /* bright yellow */
        text-align: center;
        font-size: 18px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    **Movie Genre Classification**
    
    This app predicts the genre of a movie  
    based on its plot description using:
    
    • TF-IDF  
    • Logistic Regression  
    • NLP techniques  
    """)

st.markdown("<h1 style='color:#ffff00;'>🎬 Movie Genre Predictor</h1>", unsafe_allow_html=True)
st.markdown( "<p style='color:#ffff00; text-align:center;' class='intro-text'>Enter a movie plot summary and discover its genre instantly.</p>", unsafe_allow_html=True )
plot = st.text_area(
    "📝 Movie Plot Description",
    placeholder="Type here...",
    height=180
)
st.markdown( """ <style> label[data-testid="stWidgetLabel"] { color: #ffff00 !important; /* bright yellow */ font-weight: bold; } </style> """, unsafe_allow_html=True )
if st.button("Predict Genre"):
    if plot.strip() == "":
        st.warning("⚠️ Please enter a movie description.")
    else:
        plot_tfidf = vectorizer.transform([plot])
        prediction = model.predict(plot_tfidf)

        st.markdown(
            f"""
            <div class="result-box">
                🎥 <b>Predicted Genre</b><br><br>
                <span style="color:#facc15;">{prediction[0]}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
