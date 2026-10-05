# SMS Spam Detection with Python

A machine learning project that classifies SMS messages as **SPAM** or **HAM** using `CountVectorizer` and `MultinomialNB` (Naive Bayes Algorithm).

After years developing RESTful APIs and backend systems for big corporations in Java/Springboot, I decided to get out of my comfort zone and start studying Python to improve my skills and build back my confidence on interviewing after so many years working corporate jobs. 

A second big motivation for starting this project was that I've been applying to jobs and seeing Python and Machine Learning always pop up on the roles, and I decided to learn my hand, to learn from scratch, and hopefully work on something exciting, to encourage me to learn while having fun with it. SPAM is a problem I suffer from a LOT, both in text messages and e-mails, so my goal was to build an application to simply feed from my real-life dataset and learn how to correctly identify and label potential SPAM or harmful messages.

The application is simple. One script trains the model, and the other makes predictions, based on input data it receives.

I've learned that training the model is more costly then making predictions, that Python makes it easy to implement different approaches, that I should ask myself questions about the results (for example, a poor model can score a "false" high % when the test data is unbalanced) to identify key information to use in the predictions, and learned how to compare and analyze these different approaches to finally pick the best/most optimal model for my case.

The final goal of this project will be feed the input data directly from a Gmail account, by integrating it with Gmail's API.

# How to run the app

1. Run the train.py file on the terminal 

2. Run the predict.py file on the terminal, inputting the message (String) you'd like to be analyzed

The result will be either SPAM (a SPAM message) or HAM (a legitimate message), and the confidence probability that was taken into consideration before giving the response.

# Example

Run:
``python src/predict.py 

Enter a message: We are going to a concert tonight! Pick you up at 8?  

Prediction: HAM
Ham probability:  100.00%
Spam probability: 0.00%

Enter a message: You won! Click the link to claim your free trip to Las Vegas http://link.fke.bad
Prediction: SPAM
Ham probability:  0.00%
Spam probability: 100.00%
``

