import streamlit as st
import pandas as pd
import joblib


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("career_model.pkl")


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Career Recommendation",
    page_icon="🎯",
    layout="wide"
)


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🎯 Career AI")

st.sidebar.write(
    """
    ### AI Career Path Recommendation

    This system analyzes your:

    • Academic performance  
    • Technical skills  
    • Projects  
    • Internship  
    • Communication  
    • Career interest  

    and recommends suitable career paths.
    """
)

st.sidebar.divider()

st.sidebar.info(
    "Educational project for career guidance."
)


# ==========================================
# MAIN TITLE
# ==========================================

st.title("🎯 AI Career Path Recommendation System")

st.write(
    "Discover suitable career paths based on your "
    "skills, academic performance and interests."
)

st.divider()


# ==========================================
# STUDENT PROFILE
# ==========================================

st.subheader("👤 Student Profile")

col1, col2 = st.columns(2)


# ==========================================
# COLUMN 1
# ==========================================

with col1:

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        step=0.1
    )

    python = st.selectbox(
        "🐍 Python",
        ["No", "Yes"]
    )

    sql = st.selectbox(
        "🗄️ SQL",
        ["No", "Yes"]
    )

    ml = st.selectbox(
        "🤖 Machine Learning",
        ["No", "Yes"]
    )

    powerbi = st.selectbox(
        "📊 Power BI",
        ["No", "Yes"]
    )

    java = st.selectbox(
        "☕ Java",
        ["No", "Yes"]
    )


# ==========================================
# COLUMN 2
# ==========================================

with col2:

    cloud = st.selectbox(
        "☁️ Cloud",
        ["No", "Yes"]
    )

    cybersecurity = st.selectbox(
        "🔐 Cybersecurity",
        ["No", "Yes"]
    )

    communication = st.selectbox(
        "💬 Communication Skills",
        ["No", "Yes"]
    )

    projects = st.number_input(
        "📁 Number of Projects",
        min_value=0,
        max_value=20,
        value=2
    )

    internship = st.selectbox(
        "💼 Internship Experience",
        ["No", "Yes"]
    )

    interest = st.selectbox(
        "❤️ Career Interest",
        [
            "AI",
            "Data",
            "Software",
            "Business",
            "Cloud",
            "Cybersecurity"
        ]
    )


# ==========================================
# PREDICTION BUTTON
# ==========================================

st.divider()

predict_button = st.button(
    "🚀 RECOMMEND CAREER",
    use_container_width=True
)


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    input_data = pd.DataFrame({

        "CGPA": [cgpa],

        "Python": [
            1 if python == "Yes" else 0
        ],

        "SQL": [
            1 if sql == "Yes" else 0
        ],

        "ML": [
            1 if ml == "Yes" else 0
        ],

        "PowerBI": [
            1 if powerbi == "Yes" else 0
        ],

        "Java": [
            1 if java == "Yes" else 0
        ],

        "Cloud": [
            1 if cloud == "Yes" else 0
        ],

        "Cybersecurity": [
            1 if cybersecurity == "Yes" else 0
        ],

        "Communication": [
            1 if communication == "Yes" else 0
        ],

        "Projects": [
            projects
        ],

        "Internship": [
            1 if internship == "Yes" else 0
        ],

        "Interest": [
            interest
        ]
    })


    # ======================================
    # CAREER PREDICTION
    # ======================================

    prediction = model.predict(input_data)[0]


    st.divider()

    st.subheader("🎯 Career Recommendation")

    st.success(
        f"Recommended Career: {prediction}"
    )


    # ======================================
    # CAREER PROBABILITY
    # ======================================

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_data
        )[0]

        careers = model.classes_

        results = pd.DataFrame({

            "Career": careers,

            "Match Score": probabilities * 100

        })

        results = results.sort_values(
            "Match Score",
            ascending=False
        )

        results["Match Score"] = results[
            "Match Score"
        ].round(2)


        st.subheader("📊 Career Match")

        st.bar_chart(
            results.set_index("Career")
        )

        st.dataframe(
            results,
            use_container_width=True
        )
