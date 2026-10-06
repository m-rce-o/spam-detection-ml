import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_dataset():
    print("Loading dataset...")
    try:
        df = pd.read_csv(
            "data/raw/SMSSpamCollection",
            sep="\t",
            header=None,
            names=["label", "message"]
        )

        return df
    except FileNotFoundError:
        print("Dataset not found.")
        print("Please make sure the file data/raw/SMSSpamCollection exists.")
        exit()

def main():
    df = load_dataset()

    X_text = df["message"] # The messages
    y = df["label"] # The answers, ham or spam

    print("Splitting dataset...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Vectorizing messages...")
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    print("Evaluating model and generating classification report...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\nAccuracy: {accuracy:.2%}\n")
    print(classification_report(y_test, predictions))

    print("Saving model...")
    joblib.dump(model, "model.joblib")
    joblib.dump(vectorizer, "vectorizer.joblib")

    print("Model and vectorizer saved.")

if __name__ == "__main__":
    main()