from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
postive_text = ["Love the new changes", "Big Difference", "Improved", "Great job", "Awesome", "excellent", "Exciting Changes", "Feel Supported", "Like it", "Not Bad"]
negative_text = ["Not what I expected", "Disappointed", "Worst ever", "Terrible", "Awful", "Unacceptable", "Not good", "Not happy", "Not satisfied", "Not worth it"]
training_text = postive_text + negative_text
training_labels = [1] * len(postive_text) + [0] * len(negative_text)
vectorizer = CountVectorizer()
vectorizer.fit(training_text)
training_vectors = vectorizer.transform(training_text)
classifier = DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)