import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv("career_recommendation_dataset_500.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# 2. CLEAN DATA
# ============================================================

df = df.dropna()

print("\nAfter removing missing values:")
print(df.shape)


# ============================================================
# 3. INPUT FEATURES AND TARGET
# ============================================================

X = df.drop("Career", axis=1)

y = df["Career"]


# ============================================================
# 4. CATEGORICAL FEATURE
# ============================================================

categorical_columns = ["Interest"]


# ============================================================
# 5. PREPROCESSING
# ============================================================

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


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 7. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        )
}


# ============================================================
# 8. TRAIN MODELS
# ============================================================

results = {}

trained_models = {}

print("\n")
print("=" * 50)
print("MODEL ACCURACY")
print("=" * 50)


for name, algorithm in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                algorithm
            )
        ]
    )

    # Train model
    pipeline.fit(
        X_train,
        y_train
    )

    # Prediction
    predictions = pipeline.predict(
        X_test
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    trained_models[name] = pipeline

    print(
        f"{name}: "
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_model_name = max(
    results,
    key=results.get
)

best_model = trained_models[
    best_model_name
]

best_accuracy = results[
    best_model_name
]


# ============================================================
# 10. DISPLAY BEST MODEL
# ============================================================

print("\n")
print("=" * 50)
print("BEST MODEL")
print("=" * 50)

print(
    "Best Model:",
    best_model_name
)

print(
    "Best Accuracy:",
    f"{best_accuracy * 100:.2f}%"
)


# ============================================================
# 11. DISPLAY ALL RESULTS
# ============================================================

print("\n")
print("=" * 50)
print("ALL MODEL RESULTS")
print("=" * 50)

for name, accuracy in results.items():

    print(
        f"{name:<25}"
        f"{accuracy * 100:.2f}%"
    )


# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "career_model.pkl"
)

print("\n")
print("=" * 50)
print("MODEL SAVED")
print("=" * 50)

print(
    "career_model.pkl created successfully!"
)


# ============================================================
# 13. VERIFY MODEL
# ============================================================

loaded_model = joblib.load(
    "career_model.pkl"
)

print("\nModel verification: SUCCESS")

print("\nCareer classes:")

print(
    loaded_model.classes_
)
