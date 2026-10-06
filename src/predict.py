import joblib

def load_model_from_disk():
    try:
        model = joblib.load("model.joblib")
        vectorizer = joblib.load("vectorizer.joblib")
        
        return model, vectorizer
    except FileNotFoundError:
        print("Model files not found. Please run 'python src/train.py' first.")
        exit()

def predict_message(model, vectorizer, message):
    message_vectorized = vectorizer.transform([message])
    probabilities = model.predict_proba(message_vectorized)
    
    spam_probability = probabilities[0][1]

    if spam_probability >= 0.90:
        prediction = "spam"
    else:
        prediction = "ham"
    
    return prediction, probabilities[0]

def main():
    model, vectorizer = load_model_from_disk()

    counter = 0
    while counter < 3:
        message = input("Enter a message: ").strip()
        if message:
            break
        counter += 1
        print("Please enter a message.")
        
    if counter == 3:
        print("Exiting program.")
        exit()

    prediction, probabilities = predict_message(model, vectorizer, message)

    print(f"Prediction: {prediction.upper()}")
    print(f"Ham probability:  {probabilities[0]:.2%}")
    print(f"Spam probability: {probabilities[1]:.2%}")

if __name__ == "__main__":
    main()