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

st.markdown("""
<style>

.stApp {
    background-color: #0e1726;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #151d2e;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    color: #9ca8bc;
    font-size: 16px;
    margin-bottom: 25px;
}

.card {
    background-color: #182236;
    border: 1px solid #303c52;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 18px;
}

.card-title {
    font-size: 20px;
    font-weight: 700;
    color: white;
    margin-bottom: 10px;
}

.metric-card {
    background-color: #182236;
    border: 1px solid #303c52;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    min-height: 120px;
}

.metric-number {
    font-size: 28px;
    font-weight: 800;
    color: white;
}

.metric-label {
    color: #9ca8bc;
    font-size: 14px;
    margin-top: 5px;
}

.developer-card {
    display: flex;
    align-items: center;
    gap: 22px;
    padding: 26px;
    margin: 25px 0;
    border-radius: 18px;
    background-color: #182236;
    border: 1px solid #3b4e6b;
}

.developer-icon {
    width: 75px;
    height: 75px;
    min-width: 75px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background-color: #405de6;
    font-size: 34px;
}

.developer-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #8fa5c7;
}

.developer-name {
    font-size: 30px;
    font-weight: 800;
    color: white;
    margin-top: 4px;
}

.developer-role {
    color: #aebdd3;
    font-size: 14px;
    margin-top: 4px;
}

.developer-project {
    color: #dce6f5;
    font-size: 13px;
    margin-top: 10px;
}

.prediction-card {
    background-color: #172d46;
    border: 1px solid #3c6c9f;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin: 20px 0;
}

.prediction-label {
    color: #9fb4cc;
    font-size: 13px;
    letter-spacing: 1px;
}

.prediction-career {
    color: white;
    font-size: 36px;
    font-weight: 800;
    margin: 10px 0;
}

.prediction-description {
    color: #a9bad0;
    font-size: 14px;
}

.skill-have {
    display: inline-block;
    background-color: #153d31;
    border: 1px solid #2d765e;
    color: #7ee2bf;
    padding: 7px 12px;
    border-radius: 20px;
    margin: 4px;
}

.skill-missing {
    display: inline-block;
    background-color: #402a2a;
    border: 1px solid #754848;
    color: #ffaaaa;
    padding: 7px 12px;
    border-radius: 20px;
    margin: 4px;
}

.roadmap-item {
    background-color: #182236;
    border: 1px solid #303c52;
    border-left: 4px solid #668cff;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0;
}

.roadmap-step {
    color: #7e9cff;
    font-size: 12px;
    font-weight: 800;
}

.roadmap-topic {
    color: white;
    font-size: 16px;
    font-weight: 600;
    margin-top: 4px;
}

.footer {
    text-align: center;
    color: #78859a;
    padding: 35px 0 15px 0;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATASET AND TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model():

    df = pd.read_csv(
        "career_recommendation_dataset_500.csv"
    )

    df = df.dropna()

    X = df.drop(
        "Career",
        axis=1
    )

    y = df["Career"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "interest",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                ["Interest"]
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
                "model",
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


model, model_accuracy, dataset = train_model()


# ============================================================
# CAREER SKILLS
# ============================================================

career_skills = {

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
# LEARNING ROADMAPS
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

st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        padding:10px 0 20px 0;
    ">

        <div style="font-size:42px;">
            🎯
        </div>

        <div style="
            font-size:23px;
            font-weight:800;
            color:white;
        ">
            Career AI
        </div>

        <div style="
            font-size:11px;
            color:#8fa5c7;
            margin-top:5px;
            letter-spacing:1px;
        ">
            SMART CAREER GUIDANCE
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        padding:14px;
        margin-bottom:15px;
        border-radius:12px;
        background-color:#182236;
        border:1px solid #303c52;
    ">

        <div style="font-size:25px;">
            👨‍💻
        </div>

        <b style="color:white;">
            Mayur Yadav
        </b>

        <br>

        <span style="
            font-size:11px;
            color:#9aaac0;
        ">
            AI & Data Science
        </span>

    </div>
    """,
    unsafe_allow_html=True
)


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
    **AI Career Path Recommendation System**

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


    # Developer card

    st.markdown(
        """
        <div class="developer-card">

            <div class="developer-icon">
                👨‍💻
            </div>

            <div>

                <div class="developer-label">
                    PROJECT DEVELOPED BY
                </div>

                <div class="developer-name">
                    Mayur Yadav
                </div>

                <div class="developer-role">
                    Artificial Intelligence & Data Science
                </div>

                <div class="developer-project">
                    🎯 AI Career Path Recommendation System
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Dashboard metrics

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            """
            <div class="metric-card">

                <div class="metric-number">
                    🤖
                </div>

                <div class="metric-label">
                    AI Recommendation
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {len(dataset)}
                </div>

                <div class="metric-label">
                    Dataset Records
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {len(model.classes_)}
                </div>

                <div class="metric-label">
                    Career Paths
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {model_accuracy * 100:.1f}%
                </div>

                <div class="metric-label">
                    Model Accuracy
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    # About project

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                📋 About This Project
            </div>

            <p>
            The AI Career Path Recommendation System is a
            Machine Learning based application designed to
            help students identify suitable career paths.
            </p>

            <p>
            The system analyzes CGPA, technical skills,
            projects, internship experience, communication
            skills and career interests.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # How system works

    st.subheader("✨ How This System Works")


    steps = [
        (
            "1",
            "Student Profile",
            "Enter CGPA, technical skills, projects, internship and career interest."
        ),
        (
            "2",
            "Machine Learning",
            "The Random Forest model analyzes the student's profile."
        ),
        (
            "3",
            "Career Recommendation",
            "The system predicts the most suitable career."
        ),
        (
            "4",
            "Skill Gap Analysis",
            "Identify skills that need improvement."
        ),
        (
            "5",
            "Learning Roadmap",
            "Follow a structured roadmap for the selected career."
        )
    ]


    for number, title, description in steps:

        st.markdown(
            f"""
            <div class="roadmap-item">

                <div class="roadmap-step">
                    STEP {number}
                </div>

                <div class="roadmap-topic">
                    {title}
                </div>

                <div style="
                    color:#9ca8bc;
                    margin-top:5px;
                    font-size:14px;
                ">
                    {description}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.success(
        "Select '🎯 Career Recommendation' from the sidebar to begin."
    )


# ============================================================
# CAREER RECOMMENDATION
# ============================================================

elif page == "🎯 Career Recommendation":

    st.markdown(
        '<div class="main-title">'
        '🎯 Career Recommendation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter your profile and receive an AI-powered career recommendation'
        '</div>',
        unsafe_allow_html=True
    )


    st.subheader("📚 Student Profile")


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


    st.markdown("<br>", unsafe_allow_html=True)


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


        # Prediction card

        st.markdown(
            f"""
            <div class="prediction-card">

                <div class="prediction-label">
                    AI RECOMMENDATION
                </div>

                <div class="prediction-career">
                    🎯 {prediction}
                </div>

                <div class="prediction-description">
                    Your profile shows the strongest match
                    with this career path.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # Metrics

        col1, col2, col3 = st.columns(3)


        with col1:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-number">
                        {best_score:.1f}%
                    </div>

                    <div class="metric-label">
                        Career Match
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with col2:

            technical_skills = sum([
                python == "Yes",
                sql == "Yes",
                ml == "Yes",
                powerbi == "Yes",
                java == "Yes",
                cloud == "Yes",
                cybersecurity == "Yes"
            ])


            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-number">
                        {technical_skills}/7
                    </div>

                    <div class="metric-label">
                        Technical Skills
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        with col3:

            st.markdown(
                f"""
                <div class="metric-card">

                    <div class="metric-number">
                        {projects}
                    </div>

                    <div class="metric-label">
                        Projects
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        st.markdown("<br>", unsafe_allow_html=True)


        # Match scores

        st.subheader("📊 Career Match Scores")


        chart_data = results.set_index(
            "Career"
        )["Match Score"]


        st.bar_chart(
            chart_data
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


        # Placement readiness

        st.divider()

        st.subheader(
            "🎯 Placement Readiness Score"
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
                "💻 Technical Skills",
                f"{technical_skills}/7"
            )


        with col3:

            st.metric(
                "📁 Projects",
                projects
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

    st.markdown(
        '<div class="main-title">'
        '🔍 Skill Gap Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Discover which skills you need to improve'
        '</div>',
        unsafe_allow_html=True
    )


    career = st.selectbox(
        "🎯 Select Target Career",
        list(career_skills.keys())
    )


    required_skills = career_skills[
        career
    ]


    st.subheader(
        f"📚 Required Skills — {career}"
    )


    for skill in required_skills:

        st.write(
            f"🔹 {skill}"
        )


    st.divider()


    st.subheader(
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


        percentage = (
            len(current_skills)
            / len(required_skills)
        ) * 100


        st.divider()


        st.subheader(
            "📊 Skill Match"
        )


        st.progress(
            int(percentage)
        )


        st.metric(
            "Skill Match",
            f"{percentage:.0f}%"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.markdown(
                "### ✅ Skills You Have"
            )


            if current_skills:

                for skill in current_skills:

                    st.markdown(
                        f"""
                        <span class="skill-have">
                        ✓ {skill}
                        </span>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.write(
                    "No matching skills selected."
                )


        with col2:

            st.markdown(
                "### ❌ Skills to Improve"
            )


            if missing_skills:

                for skill in missing_skills:

                    st.markdown(
                        f"""
                        <span class="skill-missing">
                        ✗ {skill}
                        </span>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.success(
                    "Excellent! You have all required skills."
                )


# ============================================================
# LEARNING ROADMAP
# ============================================================

elif page == "🗺️ Learning Roadmap":

    st.markdown(
        '<div class="main-title">'
        '🗺️ Learning Roadmap'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Follow a structured path for your target career'
        '</div>',
        unsafe_allow_html=True
    )


    career = st.selectbox(
        "🎯 Select Career",
        list(roadmaps.keys())
    )


    roadmap = roadmaps[
        career
    ]


    st.subheader(
        f"🚀 {career} Roadmap"
    )


    for i, topic in enumerate(
        roadmap,
        start=1
    ):

        st.markdown(
            f"""
            <div class="roadmap-item">

                <div class="roadmap-step">
                    STEP {i}
                </div>

                <div class="roadmap-topic">
                    {topic}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.success(
        "💡 Build practical projects while learning each skill."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":

    st.markdown(
        '<div class="main-title">'
        '📊 Model Information'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning information used by this application'
        '</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            <div class="metric-card">

                <div class="metric-number">
                    🌲
                </div>

                <div class="metric-label">
                    Random Forest
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {model_accuracy * 100:.1f}%
                </div>

                <div class="metric-label">
                    Test Accuracy
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-number">
                    {len(dataset)}
                </div>

                <div class="metric-label">
                    Dataset Records
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown("<br>", unsafe_allow_html=True)


    st.subheader(
        "📥 Input Features"
    )


    features = [
        "CGPA",
        "Python",
        "SQL",
        "Machine Learning",
        "Power BI",
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
            f"🔹 {feature}"
        )


    st.divider()


    st.subheader(
        "🎯 Career Classes"
    )


    for career in model.classes_:

        st.write(
            f"🎯 {career}"
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


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🎯 <b>AI Career Recommendation System</b>

        <br><br>

        👨‍💻 <b>Developed by Mayur Yadav</b>

        <br><br>

        Artificial Intelligence & Data Science

        <br><br>

        Powered by Machine Learning + Streamlit

        <br><br>

        Educational Project | 2026

    </div>
    """,
    unsafe_allow_html=True
)
