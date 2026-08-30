import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- APPLICATION BACKGROUND ---------- */

    .stApp {
        background-color: #f5f7fb;
    }


    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }


    /* ---------- HEADINGS ---------- */

    h1 {
        color: #111827 !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #111827 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #1f2937 !important;
        font-weight: 700 !important;
    }


    /* ---------- NORMAL TEXT ---------- */

    p {
        color: #4b5563;
    }


    /* ---------- METRIC BOX ---------- */

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 800 !important;
    }


    /* ---------- CONTAINERS ---------- */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 15px;
    }


    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }


    /* ---------- FOOTER ---------- */

    .footer-text {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET AND TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    dataset_path = "career_recommendation_dataset_500.csv"

    df = pd.read_csv(dataset_path)

    df = df.dropna()

    required_columns = [
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
        "Interest",
        "Career"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in dataset: "
            + ", ".join(missing_columns)
        )

    X = df.drop(
        "Career",
        axis=1
    )

    y = df["Career"]

    categorical_columns = [
        "Interest"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "interest",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            )
        ],
        remainder="passthrough"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                )
            )
        ]
    )

    model.fit(
        X_train,
        y_train
    )

    accuracy = model.score(
        X_test,
        y_test
    )

    return model, accuracy, df


# Train model

model, model_accuracy, dataset = train_model()


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
        "Data Visualization",
        "DAX"
    ],

    "Cybersecurity Analyst": [
        "Cybersecurity",
        "Python",
        "Networking",
        "Linux"
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
        "Machine Learning Projects"
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
        "Security Projects"
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

with st.sidebar:

    st.markdown(
        "# 🎯 Career AI"
    )

    st.caption(
        "SMART CAREER GUIDANCE"
    )

    st.divider()

    st.markdown(
        "### 👨‍💻 Developer"
    )

    st.markdown(
        "**Mayur Yadav**"
    )

    st.caption(
        "Artificial Intelligence & Data Science"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🎯 Career Recommendation",
            "🔍 Skill Gap Analysis",
            "🗺️ Learning Roadmap",
            "📊 Model Information"
        ]
    )

    st.divider()

    st.info(
        """
        **AI Career Path Recommendation System**

        The system analyzes:

        • CGPA

        • Technical skills

        • Projects

        • Internship

        • Communication

        • Career interest
        """
    )

    st.divider()

    st.caption(
        "🎓 Student Project"
    )

    st.caption(
        "Developed by Mayur Yadav"
    )


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title(
        "🎯 AI Career Path Recommendation System"
    )

    st.subheader(
        "Personalized career recommendation using Machine Learning"
    )

    st.write("")


    # --------------------------------------------------------
    # DEVELOPER
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 👨‍💻 PROJECT DEVELOPED BY"
        )

        st.markdown(
            "# Mayur Yadav"
        )

        st.write(
            "Artificial Intelligence & Data Science"
        )

        st.caption(
            "🎯 AI Career Path Recommendation System"
        )


    st.write("")


    # --------------------------------------------------------
    # DASHBOARD METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "🤖 AI System",
            "ML Powered"
        )


    with col2:

        st.metric(
            "📚 Dataset Records",
            len(dataset)
        )


    with col3:

        st.metric(
            "🎯 Career Paths",
            len(model.classes_)
        )


    with col4:

        st.metric(
            "📊 Model Accuracy",
            f"{model_accuracy * 100:.1f}%"
        )


    st.divider()


    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.header(
        "📋 About This Project"
    )

    st.write(
        """
        The AI Career Path Recommendation System is a
        Machine Learning based application designed to help
        students identify suitable career paths.

        The system analyzes academic performance, technical
        skills, projects, internship experience, communication
        skills and career interests.
        """
    )


    # --------------------------------------------------------
    # AVAILABLE CAREERS
    # --------------------------------------------------------

    st.header(
        "🚀 Available Career Paths"
    )

    careers = list(model.classes_)

    col1, col2, col3 = st.columns(3)


    for i, career in enumerate(careers):

        with [col1, col2, col3][i % 3]:

            with st.container(border=True):

                st.markdown(
                    f"### 🎯 {career}"
                )

                skills = career_skills.get(
                    career,
                    []
                )

                st.write(
                    f"**Key Skills:** {len(skills)}"
                )


    st.divider()


    # --------------------------------------------------------
    # HOW SYSTEM WORKS
    # --------------------------------------------------------

    st.header(
        "✨ How This System Works"
    )


    steps = [
        (
            "1️⃣",
            "Student Profile",
            "Enter your CGPA, technical skills, projects, internship and career interest."
        ),

        (
            "2️⃣",
            "Machine Learning",
            "The Random Forest model analyzes your profile."
        ),

        (
            "3️⃣",
            "Career Recommendation",
            "The system predicts the most suitable career."
        ),

        (
            "4️⃣",
            "Skill Gap Analysis",
            "Identify the skills you need to improve."
        ),

        (
            "5️⃣",
            "Learning Roadmap",
            "Follow a structured learning roadmap."
        )
    ]


    for icon, title, description in steps:

        with st.container(border=True):

            st.markdown(
                f"### {icon} {title}"
            )

            st.write(
                description
            )


    st.success(
        "Select '🎯 Career Recommendation' from the sidebar to begin."
    )


# ============================================================
# CAREER RECOMMENDATION
# ============================================================

elif page == "🎯 Career Recommendation":

    st.title(
        "🎯 Career Recommendation"
    )

    st.subheader(
        "Enter your profile to receive an AI-based career recommendation"
    )


    st.divider()


    # --------------------------------------------------------
    # STUDENT PROFILE
    # --------------------------------------------------------

    st.header(
        "📚 Student Profile"
    )


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
            value=2,
            step=1
        )

        internship = st.selectbox(
            "💼 Internship",
            ["No", "Yes"]
        )


        interests = sorted(
            dataset["Interest"]
            .astype(str)
            .unique()
            .tolist()
        )


        interest = st.selectbox(
            "❤️ Career Interest",
            interests
        )


    st.write("")


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🚀 RECOMMEND MY CAREER",
        use_container_width=True,
        type="primary"
    )


    if predict_button:

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "CGPA": [
                cgpa
            ],

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
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


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


        best_score = results.iloc[0]["Match Score"]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.header(
            "📋 Prediction Result"
        )


        with st.container(border=True):

            st.success(
                f"🎯 Recommended Career: {prediction}"
            )

            st.markdown(
                f"# {prediction}"
            )

            st.write(
                f"Career Match Score: **{best_score:.1f}%**"
            )


        st.write("")


        # ----------------------------------------------------
        # RESULT METRICS
        # ----------------------------------------------------

        technical_skills = sum([
            python == "Yes",
            sql == "Yes",
            ml == "Yes",
            powerbi == "Yes",
            java == "Yes",
            cloud == "Yes",
            cybersecurity == "Yes"
        ])


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "🎯 Career Match",
                f"{best_score:.1f}%"
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


        st.divider()


        # ----------------------------------------------------
        # CAREER MATCH CHART
        # ----------------------------------------------------

        st.header(
            "📊 Career Match Scores"
        )


        chart_data = results.set_index(
            "Career"
        )["Match Score"]


        st.bar_chart(
            chart_data
        )


        # ----------------------------------------------------
        # CAREER COMPARISON
        # ----------------------------------------------------

        st.subheader(
            "📋 Detailed Career Comparison"
        )


        display_results = results.copy()

        display_results["Match Score"] = (
            display_results["Match Score"]
            .round(2)
        )


        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # RECOMMENDED SKILLS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            f"📚 Important Skills for {prediction}"
        )


        recommended_skills = career_skills.get(
            prediction,
            []
        )


        skill_col1, skill_col2, skill_col3 = st.columns(3)


        for i, skill in enumerate(
            recommended_skills
        ):

            with [
                skill_col1,
                skill_col2,
                skill_col3
            ][i % 3]:

                st.info(
                    f"💡 {skill}"
                )


        # ----------------------------------------------------
        # PLACEMENT READINESS
        # ----------------------------------------------------

        st.divider()

        st.header(
            "🎯 Placement Readiness"
        )


        technical_score = (
            technical_skills / 7
        ) * 40


        project_score = min(
            projects / 5,
            1
        ) * 20


        internship_score = (
            15
            if internship == "Yes"
            else 0
        )


        communication_score = (
            15
            if communication == "Yes"
            else 0
        )


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


        readiness = min(
            readiness,
            100
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "🎯 Readiness",
                f"{readiness:.0f}/100"
            )


        with col2:

            st.metric(
                "💼 Internship",
                internship
            )


        with col3:

            st.metric(
                "💬 Communication",
                communication
            )


        st.progress(
            int(readiness)
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

    st.title(
        "🔍 Skill Gap Analysis"
    )

    st.subheader(
        "Identify the skills you need to improve"
    )


    career = st.selectbox(
        "🎯 Select Target Career",
        list(career_skills.keys())
    )


    required_skills = career_skills[
        career
    ]


    st.divider()


    st.header(
        f"📚 Required Skills for {career}"
    )


    skill_columns = st.columns(3)


    for i, skill in enumerate(
        required_skills
    ):

        with skill_columns[i % 3]:

            st.info(
                f"📌 {skill}"
            )


    st.divider()


    st.header(
        "👤 Your Current Skills"
    )


    skill_status = {}


    for skill in required_skills:

        skill_status[skill] = st.selectbox(
            f"Do you know {skill}?",
            ["No", "Yes"],
            key=f"{career}_{skill}"
        )


    analyze_button = st.button(
        "🔍 ANALYZE MY SKILL GAP",
        use_container_width=True,
        type="primary"
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


        percentage = (
            len(current_skills)
            / len(required_skills)
        ) * 100


        st.divider()


        st.header(
            "📊 Skill Match"
        )


        st.progress(
            int(percentage)
        )


        st.metric(
            "Current Skill Match",
            f"{percentage:.0f}%"
        )


        st.write("")


        col1, col2 = st.columns(2)


        with col1:

            st.subheader(
                "✅ Skills You Have"
            )


            if current_skills:

                for skill in current_skills:

                    st.success(
                        f"✓ {skill}"
                    )

            else:

                st.info(
                    "No matching skills selected."
                )


        with col2:

            st.subheader(
                "📚 Skills to Improve"
            )


            if missing_skills:

                for skill in missing_skills:

                    st.warning(
                        f"→ {skill}"
                    )

            else:

                st.success(
                    "Excellent! You have all required skills."
                )


# ============================================================
# LEARNING ROADMAP
# ============================================================

elif page == "🗺️ Learning Roadmap":

    st.title(
        "🗺️ Learning Roadmap"
    )

    st.subheader(
        "Follow a structured path for your target career"
    )


    career = st.selectbox(
        "🎯 Select Career",
        list(roadmaps.keys())
    )


    roadmap = roadmaps[
        career
    ]


    st.divider()


    st.header(
        f"🚀 {career} Learning Roadmap"
    )


    st.write(
        f"Total learning stages: **{len(roadmap)}**"
    )


    for i, topic in enumerate(
        roadmap,
        start=1
    ):

        with st.container(border=True):

            st.markdown(
                f"### Step {i}: {topic}"
            )

            if i == 1:

                st.caption(
                    "Start with the fundamentals."
                )

            elif i == len(roadmap):

                st.caption(
                    "Build a practical project using your knowledge."
                )

            else:

                st.caption(
                    "Learn this skill and practice it with examples."
                )


    st.success(
        "💡 Tip: Build practical projects while following the roadmap."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":

    st.title(
        "📊 Model Information"
    )

    st.subheader(
        "Machine Learning information used by this application"
    )


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🤖 Algorithm",
            "Random Forest"
        )


    with col2:

        st.metric(
            "📊 Test Accuracy",
            f"{model_accuracy * 100:.1f}%"
        )


    with col3:

        st.metric(
            "📚 Dataset Size",
            len(dataset)
        )


    st.divider()


    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.header(
        "📋 Dataset Information"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Records",
            len(dataset)
        )


    with col2:

        st.metric(
            "Features",
            len(dataset.columns) - 1
        )


    with col3:

        st.metric(
            "Career Classes",
            len(model.classes_)
        )


    st.divider()


    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------

    st.header(
        "📥 Input Features"
    )


    feature_descriptions = {

        "CGPA":
            "Academic performance of the student.",

        "Python":
            "Python programming skill.",

        "SQL":
            "SQL and database skill.",

        "ML":
            "Machine Learning knowledge.",

        "PowerBI":
            "Power BI and business intelligence skill.",

        "Java":
            "Java programming skill.",

        "Cloud":
            "Cloud computing knowledge.",

        "Cybersecurity":
            "Cybersecurity knowledge.",

        "Communication":
            "Communication skill.",

        "Projects":
            "Number of completed projects.",

        "Internship":
            "Internship experience.",

        "Interest":
            "Student's career interest."
    }


    for feature, description in feature_descriptions.items():

        with st.container(border=True):

            st.markdown(
                f"### 🔹 {feature}"
            )

            st.caption(
                description
            )


    st.divider()


    # --------------------------------------------------------
    # CAREER CLASSES
    # --------------------------------------------------------

    st.header(
        "🎯 Career Classes"
    )


    career_columns = st.columns(3)


    for i, career in enumerate(
        model.classes_
    ):

        with career_columns[i % 3]:

            st.info(
                f"🎯 {career}"
            )


    st.divider()


    # --------------------------------------------------------
    # MODEL EXPLANATION
    # --------------------------------------------------------

    st.header(
        "🤖 How Random Forest Works"
    )


    st.write(
        """
        Random Forest is an ensemble Machine Learning
        algorithm that combines multiple decision trees.

        Each decision tree makes a prediction. The Random
        Forest combines these predictions to produce the
        final classification result.

        In this project, Random Forest is used to classify
        a student's profile into a suitable career path.
        """
    )


    st.divider()


    st.warning(
        """
        ⚠️ Disclaimer:

        This application is an educational project.
        Career recommendations are intended for guidance
        and demonstration purposes only. They do not
        guarantee employment or career success.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer-text">
        🎯 AI Career Recommendation System
        <br><br>
        👨‍💻 Developed by Mayur Yadav
        <br>
        Artificial Intelligence & Data Science
        <br><br>
        Powered by Machine Learning + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
