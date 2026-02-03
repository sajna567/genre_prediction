import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train_df = pd.read_csv(
    "data/train_data.txt",
    sep=":::",
    engine="python",
    names=["id", "title", "genre", "description"]
)
train_df.dropna(inplace=True)
train_df["genre"] = train_df["genre"].apply(lambda x: x.split("|")[0])

X = train_df["description"]
y = train_df["genre"]

# TF-IDF VECTORIZATION

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(X)

# TRAIN MODEL

model = LogisticRegression(
    max_iter=300,
    class_weight="balanced"
)

model.fit(X_tfidf, y)

# SAVE MODEL

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# EVALUATION (OPTIONAL)

test_sol = pd.read_csv(
    "data/test_data_solution.txt",
    sep=":::",
    engine="python",
    names=["id", "title", "genre", "description"]
)

test_sol.dropna(inplace=True)
test_sol["genre"] = test_sol["genre"].apply(lambda x: x.split("|")[0])

X_test_tfidf = vectorizer.transform(test_sol["description"])
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(test_sol["genre"], predictions)
print("Model trained successfully")
print("Test Accuracy:", accuracy)
