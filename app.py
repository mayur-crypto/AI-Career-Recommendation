import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("career_model.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎯",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CAREER SKILLS
# ============================================================

career_skills = {

    "BI Developer": [
        "SQL",
        "Power BI",
        "Data Visualization"
    ],

    "Cybersecurity Analyst": [
        "Cybersecurity",
        "Python",
        "Networking"
    ],

    "Data Analyst": [
        "Python",
        "SQL",
        "Power BI",
        "Statistics"
    ],

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

    "Software Developer": [
        "Java",
        "Python",
        "SQL",
        "Data Structures"
    ]
}

# ============================================================
# LEARNING ROADMAP
# ============================================================

roadmaps = {

    "BI Developer": [
        "SQL",
        "Excel",
        "Power BI",
        "DAX",
        "Data Modeling",
        "Dashboard Development",
        "BI Projects"
    ],

    "Cybersecurity Analyst": [
        "Computer Networks",
        "Linux",
        "Cybersecurity Fundamentals",
        "Python",
        "Security Tools",
        "Ethical Security Concepts",
        "Security Projects"
    ],

    "Data Analyst": [
        "Excel",
        "SQL",
        "Python",
        "Pandas",
        "Statistics",
        "Power BI",
        "Data Visualization",
        "Data Analysis Projects"
    ],

    "Data Scientist": [
        "Python",
        "NumPy and Pandas",
        "SQL",
        "Statistics",
        "Machine Learning",
        "Data Visualization",
        "Deep Learning",
        "Data Science Projects"
    ],

    "ML Engineer": [
        "Python",
        "NumPy and Pandas",
        "Machine Learning",
        "Deep Learning",
        "APIs",
        "Cloud Computing",
        "MLOps",
        "ML Projects"
    ],

    "Software Developer": [
        "Programming Fundamentals",
        "Python / Java",
        "Data Structures",
        "Algorithms",
        "SQL",
        "Git and GitHub",
        "Software Development",
        "Software Projects"
    ]
}

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎯 Career AI")

page = st.sidebar.radio(
    "Navigation",
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
    AI Career Path Recommendation System

    This application analyzes student
    academic performance, technical skills,
    projects, internship experience,
    communication skills and interests.
    """
)


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
        'Personalized career recommendation using Machine Learning'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🤖 AI",
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

    st.subheader("✨ How This System Works")

    st.write(
        """
        **Step 1 — Student Profile**

        Enter your CGPA, technical skills,
        projects, internship and career interest.

        **Step 2 — Machine Learning**

        The trained ML model analyzes your profile.

        **Step 3 — Career Recommendation**

        The system recommends suitable career paths.

        **Step 4 — Skill Gap Analysis**

        Identify the skills you need to improve.

        **Step 5 — Learning Roadmap**

        Follow a career-specific learning roadmap.
        """
    )

    st.success(
        "Select '🎯 Career Recommendation' from the sidebar "
        "to begin."
    )


# ============================================================
# CAREER RECOMMENDATION PAGE
# ============================================================

elif page == "🎯 Career Recommendation":

    st.title("🎯 Career Recommendation")

    st.write(
        "Enter your academic performance, skills and interests."
    )

    st.divider()

    # --------------------------------------------------------
    # STUDENT INPUT
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

    predict_button = st.button(
        "🚀 RECOMMEND MY CAREER",
        use_container_width=True
    )


    # ========================================================
    # PREDICTION
    # ========================================================

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


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(
            input_data
        )[0]

        st.divider()

        st.subheader("🏆 Recommended Career")

        st.success(
            f"🎯 {prediction}"
        )


        # ====================================================
        # CAREER MATCH SCORE
        # ====================================================

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

            st.subheader(
                "📊 Career Match Scores"
            )

            st.bar_chart(
                results.set_index("Career")[
                    "Match Score (%)"
                ]
            )

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # PLACEMENT READINESS
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 Placement Readiness Score"
        )

        # Technical skills
        technical_skills = sum([
            python == "Yes",
            sql == "Yes",
            ml == "Yes",
            powerbi == "Yes",
            java == "Yes",
            cloud == "Yes",
            cybersecurity == "Yes"
        ])

        technical_score = (
            technical_skills / 7
        ) * 40

        # Projects
        project_score = min(
            projects / 5,
            1
        ) * 20

        # Internship
        internship_score = (
            15 if internship == "Yes"
            else 0
        )

        # Communication
        communication_score = (
            15 if communication == "Yes"
            else 0
        )

        # CGPA
        cgpa_score = (
            cgpa / 10
        ) * 10

        readiness = (
            technical_score
            + project_score
            + internship_score
            + communication_score
            + cgpa_score
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🎯 Readiness",
                f"{readiness:.0f}/100"
            )

        with col2:

            st.metric(
                "💻 Technical Skills",
                f"{technical_skills}/7"
            )

        with col3:

            st.metric(
                "📁 Projects",
                projects
            )

        if readiness >= 80:

            st.success(
                "🌟 Excellent placement readiness!"
            )

        elif readiness >= 60:

            st.info(
                "👍 Good readiness. Continue improving your skills."
            )

        else:

            st.warning(
                "📚 More preparation is recommended."
            )


# ============================================================
# SKILL GAP ANALYSIS
# ============================================================

elif page == "🔍 Skill Gap Analysis":

    st.title("🔍 Skill Gap Analysis")

    st.write(
        "Find out which skills you need for your target career."
    )

    career = st.selectbox(
        "🎯 Select Career",
        list(career_skills.keys())
    )

    required_skills = career_skills[
        career
    ]

    st.subheader(
        f"📚 Skills Required for {career}"
    )

    for skill in required_skills:

        st.write(
            f"🔹 {skill}"
        )

    st.divider()

    st.subheader(
        "👤 Enter Your Current Skills"
    )

    skill_status = {}

    for skill in required_skills:

        skill_status[skill] = st.selectbox(
            f"Do you know {skill}?",
            ["No", "Yes"],
            key=f"skill_{career}_{skill}"
        )

    analyze_button = st.button(
        "🔍 ANALYZE MY SKILL GAP",
        use_container_width=True
    )

    if analyze_button:

        current_skills = []

        missing_skills = []

        for skill in required_skills:

            if skill_status[skill] == "Yes":

                current_skills.append(
                    skill
                )

            else:

                missing_skills.append(
                    skill
                )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "✅ Skills You Have"
            )

            if current_skills:

                for skill in current_skills:

                    st.write(
                        f"✅ {skill}"
                    )

            else:

                st.write(
                    "No matching skills selected."
                )

        with col2:

            st.subheader(
                "❌ Skills to Improve"
            )

            if missing_skills:

                for skill in missing_skills:

                    st.write(
                        f"❌ {skill}"
                    )

            else:

                st.write(
                    "No missing skills!"
                )

        st.divider()

        total = len(required_skills)

        acquired = len(current_skills)

        percentage = (
            acquired / total
        ) * 100

        st.metric(
            "Skill Match",
            f"{percentage:.0f}%"
        )

        if percentage >= 80:

            st.success(
                "🌟 Excellent skill match!"
            )

        elif percentage >= 50:

            st.info(
                "👍 Good start. Work on the missing skills."
            )

        else:

            st.warning(
                "📚 You should focus on developing the required skills."
            )


# ============================================================
# LEARNING ROADMAP
# ============================================================

elif page == "🗺️ Learning Roadmap":

    st.title(
        "🗺️ Personalized Learning Roadmap"
    )

    st.write(
        "Follow these steps to prepare for your selected career."
    )

    career = st.selectbox(
        "🎯 Select Career",
        list(roadmaps.keys())
    )

    roadmap = roadmaps[
        career
    ]

    st.subheader(
        f"🚀 Roadmap for {career}"
    )

    for i, skill in enumerate(
        roadmap,
        start=1
    ):

        st.write(
            f"### {i}. {skill}"
        )

        if i < len(roadmap):

            st.write("⬇️")

    st.divider()

    st.success(
        "💡 Build projects while learning each skill."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":

    st.title(
        "📊 Model Information"
    )

    st.subheader(
        "🤖 Machine Learning"
    )

    st.write(
        """
        The application uses a trained Machine Learning
        classification model to recommend a suitable
        career based on the student's profile.
        """
    )

    st.divider()

    st.subheader(
        "📥 Input Features"
    )

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

    st.subheader(
        "🎯 Target Variable"
    )

    st.info(
        "Career"
    )

    st.write(
        """
        The model predicts the Career category
        using the student's profile.
        """
    )

    st.divider()

    st.subheader(
        "⚠️ Disclaimer"
    )

    st.warning(
        """
        This application is an educational project.
        Career recommendations are guidance and are
        not a guarantee of employment or career success.
        """
    )
