import joblib

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

#message = "Congratulations! You have won a free vacation. Claim your prize now!"
message = input("Enter a message: ")

message_vectorized = vectorizer.transform([message])

probabilities = model.predict_proba(message_vectorized)
spam_probability = probabilities[0][1]

if spam_probability >= 0.90:
    prediction = "spam"
else:
    prediction = "ham"

print(f"Prediction: {prediction.upper()}")
print(f"Ham probability:  {probabilities[0][0]:.2%}")
print(f"Spam probability: {spam_probability:.2%}")