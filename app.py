import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("career_model.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Career Recommendation",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #ddd;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎯 Career AI")

st.sidebar.write("### Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "🎯 Career Recommendation",
        "🔍 Skill Gap Analysis",
        "🗺️ Learning Roadmap",
        "📊 Model Information"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    This application uses Machine Learning
    to recommend suitable career paths based
    on student skills, academic performance,
    projects, internship and interests.
    """
)


# ============================================================
# CAREER SKILLS
# ============================================================

career_skills = {

    "Data Scientist": [
        "Python",
        "SQL",
        "Machine Learning",
        "Statistics",
        "Data Visualization"
    ],

    "ML Engineer": [
        "Python",
        "Machine Learning",
        "Cloud",
        "Deep Learning"
    ],

    "Data Analyst": [
        "Python",
        "SQL",
        "Power BI",
        "Statistics"
    ],

    "BI Developer": [
        "SQL",
        "Power BI",
        "Data Visualization"
    ],

    "Software Developer": [
        "Java",
        "Python",
        "SQL"
    ],

    "Cloud Engineer": [
        "Cloud",
        "Python",
        "Linux"
    ],

    "Cybersecurity Analyst": [
        "Cybersecurity",
        "Python",
        "Networking"
    ],

    "AI Engineer": [
        "Python",
        "Machine Learning",
        "Deep Learning",
        "Cloud"
    ]
}


# ============================================================
# LEARNING ROADMAP
# ============================================================

roadmaps = {

    "Data Scientist": [
        "Python",
        "NumPy and Pandas",
        "SQL",
        "Statistics",
        "Machine Learning",
        "Data Visualization",
        "Deep Learning",
        "Real-world Projects"
    ],

    "ML Engineer": [
        "Python",
        "NumPy and Pandas",
        "Machine Learning",
        "Deep Learning",
        "APIs",
        "Cloud Computing",
        "MLOps",
        "Real-world Projects"
    ],

    "Data Analyst": [
        "Excel",
        "SQL",
        "Python",
        "Pandas",
        "Statistics",
        "Power BI",
        "Data Visualization",
        "Projects"
    ],

    "BI Developer": [
        "Excel",
        "SQL",
        "Database",
        "Power BI",
        "DAX",
        "Data Modeling",
        "Dashboard Projects"
    ],

    "Software Developer": [
        "Programming Fundamentals",
        "Java",
        "Data Structures",
        "Algorithms",
        "SQL",
        "Git and GitHub",
        "Software Projects"
    ],

    "Cloud Engineer": [
        "Python",
        "Linux",
        "Networking",
        "Cloud Fundamentals",
        "AWS/Azure/GCP",
        "Docker",
        "DevOps"
    ],

    "Cybersecurity Analyst": [
        "Networking",
        "Linux",
        "Cybersecurity Fundamentals",
        "Python",
        "Ethical Security Concepts",
        "Security Tools",
        "Security Projects"
    ],

    "AI Engineer": [
        "Python",
        "NumPy and Pandas",
        "Machine Learning",
        "Deep Learning",
        "NLP",
        "Computer Vision",
        "Cloud",
        "AI Projects"
    ]
}


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        '🎯 AI Career Path Recommendation System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Find the career path that best matches your skills '
        'and interests.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🤖 AI/ML",
            "Career Prediction"
        )

    with col2:
        st.metric(
            "🔍 Skill",
            "Gap Analysis"
        )

    with col3:
        st.metric(
            "🗺️ Learning",
            "Roadmap"
        )

    st.divider()

    st.subheader("✨ How It Works")

    st.write("""
    **1️⃣ Enter your profile**

    Provide your CGPA, technical skills,
    projects, internship experience and career interest.

    **2️⃣ AI analyzes your profile**

    The trained Machine Learning model evaluates
    your profile.

    **3️⃣ Get career recommendations**

    The system recommends suitable career paths.

    **4️⃣ Identify your skill gap**

    See which skills you need to improve.

    **5️⃣ Follow your learning roadmap**

    Get a structured roadmap for your recommended career.
    """)

    st.success(
        "Go to '🎯 Career Recommendation' from the sidebar "
        "to get started."
    )


# ============================================================
# CAREER RECOMMENDATION PAGE
# ============================================================

elif page == "🎯 Career Recommendation":

    st.title("🎯 Career Recommendation")

    st.write(
        "Enter your academic and technical information."
    )

    st.divider()

    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        cgpa = st.number_input(
            "📚 CGPA",
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
            "💬 Communication",
            ["No", "Yes"]
        )

        projects = st.number_input(
            "📁 Number of Projects",
            min_value=0,
            max_value=20,
            value=2
        )

        internship = st.selectbox(
            "💼 Internship",
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

    st.divider()

    predict = st.button(
        "🚀 RECOMMEND MY CAREER",
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict:

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

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]

        st.divider()

        st.subheader("🏆 Your Recommended Career")

        st.success(
            f"🎯 {prediction}"
        )

        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            careers = model.classes_

            results = pd.DataFrame({

                "Career": careers,

                "Match Score (%)": probabilities * 100

            })

            results = results.sort_values(
                "Match Score (%)",
                ascending=False
            )

            results["Match Score (%)"] = results[
                "Match Score (%)"
            ].round(2)

            st.subheader("📊 Career Match Scores")

            st.bar_chart(
                results.set_index("Career")
            )

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

elif page == "🔍 Skill Gap Analysis":

    st.title("🔍 Skill Gap Analysis")

    st.write(
        "Select your recommended career to see "
        "which skills you should develop."
    )

    career = st.selectbox(
        "Select Career",
        list(career_skills.keys())
    )

    required_skills = career_skills[career]

    st.subheader(
        f"🎯 Skills Required for {career}"
    )

    for skill in required_skills:

        st.write(
            f"🔹 {skill}"
        )

    st.divider()

    st.subheader("👤 Enter Your Current Skills")

    skill_status = {}

    for skill in required_skills:

        skill_status[skill] = st.selectbox(
            f"Do you know {skill}?",
            ["No", "Yes"],
            key=f"skill_{skill}"
        )

    if st.button(
        "🔍 Analyze Skill Gap",
        use_container_width=True
    ):

        current_skills = []

        missing_skills = []

        for skill in required_skills:

            if skill_status[skill] == "Yes":

                current_skills.append(skill)

            else:

                missing_skills.append(skill)

        st.divider()

        st.subheader("📋 Analysis Result")

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f"✅ Skills You Have: "
                f"{len(current_skills)}"
            )

            for skill in current_skills:

                st.write(
                    f"✅ {skill}"
                )

        with col2:

            st.error(
                f"❌ Skills to Improve: "
                f"{len(missing_skills)}"
            )

            for skill in missing_skills:

                st.write(
                    f"❌ {skill}"
                )

        if missing_skills:

            st.warning(
                "Focus on the missing skills to "
                "improve your career readiness."
            )

        else:

            st.success(
                "🎉 You have all the listed skills "
                "for this career!"
            )


# ============================================================
# LEARNING ROADMAP
# ============================================================

elif page == "🗺️ Learning Roadmap":

    st.title("🗺️ Personalized Learning Roadmap")

    career = st.selectbox(
        "Select Your Career",
        list(roadmaps.keys())
    )

    st.subheader(
        f"🚀 Roadmap for {career}"
    )

    roadmap = roadmaps[career]

    for i, skill in enumerate(
        roadmap,
        start=1
    ):

        st.write(
            f"### {i}. {skill}"
        )

        if i < len(roadmap):

            st.write("⬇️")

    st.success(
        "Complete these steps one by one and "
        "build projects alongside your learning."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":

    st.title("📊 Model Information")

    st.subheader("🤖 Machine Learning Model")

    st.write(
        "The application uses the trained Machine Learning "
        "model stored in `career_model.pkl`."
    )

    st.divider()

    st.subheader("📥 Input Features")

    features = [
        "CGPA",
        "Python",
        "SQL",
        "ML",
        "PowerBI",
        "Java",
        "Cloud",
        "Cybersecurity",
        "Communication",
        "Projects",
        "Internship",
        "Interest"
    ]

    for feature in features:

        st.write(
            f"• {feature}"
        )

    st.divider()

    st.subheader("🎯 Target Variable")

    st.info(
        "Career"
    )

    st.write(
        "The model predicts the Career category "
        "based on the student's profile."
    )

    st.divider()

    st.subheader("⚠️ Important Note")

    st.warning(
        "This application is an educational project. "
        "Career recommendations should be treated as "
        "guidance rather than a guaranteed prediction "
        "of future employment."
    )
