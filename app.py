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

    /* Main application */

    .stApp {
        background-color: #0e1726;
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        background-color: #151d2e;
    }


    /* Main title */

    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
    }


    .subtitle {
        color: #9ca8bc;
        font-size: 17px;
        margin-bottom: 25px;
    }


    /* Developer card */

    .developer-card {
        background-color: #182236;
        border: 1px solid #3b4e6b;
        border-radius: 18px;
        padding: 25px;
        margin: 20px 0;
    }


    .developer-label {
        color: #8fa5c7;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    }


    .developer-name {
        color: white;
        font-size: 32px;
        font-weight: 800;
        margin-top: 5px;
    }


    .developer-role {
        color: #aebdd3;
        font-size: 15px;
        margin-top: 5px;
    }


    .developer-project {
        color: #dce6f5;
        font-size: 14px;
        margin-top: 12px;
    }


    /* Prediction */

    .prediction-card {
        background-color: #172d46;
        border: 1px solid #3c6c9f;
        border-radius: 18px;
        padding: 30px;
        text-align: center;
        margin: 25px 0;
    }


    .prediction-label {
        color: #9fb4cc;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
    }


    .prediction-career {
        color: white;
        font-size: 38px;
        font-weight: 800;
        margin: 10px 0;
    }


    .prediction-description {
        color: #a9bad0;
        font-size: 14px;
    }


    /* Footer */

    .footer {
        text-align: center;
        color: #78859a;
        padding: 35px 0 15px 0;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET + MODEL
# ============================================================

@st.cache_resource
def train_model():

    # Load dataset
    df = pd.read_csv(
        "career_recommendation_dataset_500.csv"
    )

    # Remove empty rows
    df = df.dropna()

    # Separate input and target
    X = df.drop(
        "Career",
        axis=1
    )

    y = df["Career"]

    # Categorical column
    categorical_columns = [
        "Interest"
    ]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            )
        ],
        remainder="passthrough"
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Random Forest model
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

    # Train
    model.fit(
        X_train,
        y_train
    )

    # Accuracy
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
        "ML Projects"
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

    st.title("🎯 Career AI")

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


    # Developer section

    with st.container(border=True):

        st.markdown(
            "### 👨‍💻 Project Developed By"
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


    # Dashboard metrics

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


    # About

    st.subheader(
        "📋 About This Project"
    )

    st.write(
        """
        The AI Career Path Recommendation System is a
        Machine Learning based application designed to
        help students identify suitable career paths.

        It analyzes academic performance, technical skills,
        projects, internship experience, communication skills
        and career interests to recommend a suitable career.
        """
    )


    # Career paths

    st.subheader(
        "🚀 Available Career Paths"
    )

    careers = list(model.classes_)

    columns = st.columns(3)

    for i, career in enumerate(careers):

        with columns[i % 3]:

            with st.container(border=True):

                st.markdown(
                    f"### 🎯 {career}"
                )

                required = career_skills.get(
                    career,
                    []
                )

                st.caption(
                    f"{len(required)} key skills"
                )


    st.divider()


    # How system works

    st.subheader(
        "✨ How This System Works"
    )

    steps = [
        (
            "1️⃣",
            "Student Profile",
            "Enter CGPA, technical skills, projects, internship and career interest."
        ),

        (
            "2️⃣",
            "Machine Learning",
            "Random Forest analyzes the student's profile."
        ),

        (
            "3️⃣",
            "Career Recommendation",
            "The model predicts the most suitable career."
        ),

        (
            "4️⃣",
            "Skill Gap Analysis",
            "Identify skills that need improvement."
        ),

        (
            "5️⃣",
            "Learning Roadmap",
            "Follow a structured path for the selected career."
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
        "Select '🎯 Career Recommendation' from the sidebar to start."
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
        'Enter your profile to receive an AI-based recommendation'
        '</div>',
        unsafe_allow_html=True
    )


    st.subheader(
        "📚 Student Academic Profile"
    )


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

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


        # Get actual interests from dataset

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


    predict = st.button(
        "🚀 RECOMMEND MY CAREER",
        use_container_width=True,
        type="primary"
    )


    if predict:

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
        # RESULT CARD
        # ----------------------------------------------------

        with st.container(border=True):

            st.markdown(
                "### 📋 AI RECOMMENDATION"
            )

            st.markdown(
                f"# 🎯 {prediction}"
            )

            st.write(
                "Your profile shows the strongest match "
                "with this career path."
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

        st.subheader(
            "📊 Career Match Scores"
        )


        chart_data = results.set_index(
            "Career"
        )["Match Score"]


        st.bar_chart(
            chart_data
        )


        # ----------------------------------------------------
        # CAREER TABLE
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


        skill_columns = st.columns(3)


        for i, skill in enumerate(
            recommended_skills
        ):

            with skill_columns[i % 3]:

                st.info(
                    f"💡 {skill}"
                )


        # ----------------------------------------------------
        # PLACEMENT READINESS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
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

    st.markdown(
        '<div class="main-title">'
        '🔍 Skill Gap Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Identify the skills you need to improve for your target career'
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
            key=f"skill_{career}_{skill}"
        )


    analyze = st.button(
        "🔍 ANALYZE SKILL GAP",
        use_container_width=True,
        type="primary"
    )


    if analyze:

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
            "📊 Your Skill Match"
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

                    st.success(
                        f"✓ {skill}"
                    )

            else:

                st.write(
                    "No matching skills selected."
                )


        with col2:

            st.markdown(
                "### 📚 Skills to Improve"
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

    st.markdown(
        '<div class="main-title">'
        '🗺️ Learning Roadmap'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Follow a structured learning path for your target career'
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


    st.write(
        f"{len(roadmap)} learning stages"
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
                    "Build a practical project using this knowledge."
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


    # Model metrics

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


    # Dataset information

    st.subheader(
        "📋 Dataset Information"
    )


    st.write(
        f"Number of records: **{len(dataset)}**"
    )


    st.write(
        f"Number of features: **{len(dataset.columns) - 1}**"
    )


    st.write(
        f"Number of career classes: **{len(model.classes_)}**"
    )


    st.divider()


    # Features

    st.subheader(
        "📥 Input Features"
    )


    feature_descriptions = {

        "CGPA":
            "Academic performance of the student.",

        "Python":
            "Python programming skill.",

        "SQL":
            "SQL/database skill.",

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


    # Career classes

    st.subheader(
        "🎯 Career Classes"
    )


    for career in model.classes_:

        st.write(
            f"🎯 {career}"
        )


    st.divider()


    # Model explanation

    st.subheader(
        "🤖 How Random Forest Works"
    )


    st.write(
        """
        Random Forest is an ensemble Machine Learning
        algorithm that combines multiple decision trees.

        Each tree makes a prediction and the final result
        is determined by combining the predictions of the
        individual trees.

        In this project, Random Forest is used as a
        classification algorithm to predict the most
        suitable career path for a student.
        """
    )


    st.divider()


    st.warning(
        """
        ⚠️ Disclaimer:

        This application is an educational project.
        The recommendation is intended for guidance and
        demonstration purposes only. It does not guarantee
        employment or career success.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎯 AI Career Recommendation System"
)

st.caption(
    "👨‍💻 Developed by Mayur Yadav | Artificial Intelligence & Data Science"
)

st.caption(
    "Powered by Machine Learning + Streamlit | Educational Project"
)
