import xml.etree.ElementTree as ET
import nltk
import pandas as pd
from nltk.corpus import sentiwordnet as swn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

def parse_and_combine_xml(files):
    combined_df = pd.DataFrame()
    for file_path in files:
        tree = ET.parse(file_path)
        root = tree.getroot()
        rows = []

        for sentence in root.findall('.//sentence'):
            text = sentence.find('text').text
            aspect_terms = sentence.find('aspectTerms')

            if aspect_terms is not None:
                for at in aspect_terms.findall('aspectTerm'):
                    term = at.get('term')
                    polarity = at.get('polarity')
                    if polarity != 'conflict':  # Directly exclude 'conflict' polarities here
                        from_idx = int(at.get('from'))
                        to_idx = int(at.get('to'))
                        rows.append({'text': text, 'aspect_term': term, 'polarity': polarity, 'from': from_idx, 'to': to_idx})

        temp_df = pd.DataFrame(rows)
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
    return combined_df

train_files = [
    'SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train.xml',
    'SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train.xml'
]

train_df = parse_and_combine_xml(train_files)

# Preprocessing and feature extraction
stop_words_list = list(stopwords.words('english'))
vectorizer = TfidfVectorizer(stop_words=stop_words_list)

# Prepare feature matrix and labels
X = vectorizer.fit_transform(train_df['text'])
y = train_df['polarity'].map({'positive': 1, 'negative': 0, 'neutral': 2}).astype(int)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate on the validation set
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# accuracy obtained :
# precision    recall  f1-score   support
#
#           0       0.65      0.59      0.62       326
#           1       0.77      0.89      0.82       641
#           2       0.56      0.37      0.44       216
# 
#    accuracy                           0.71      1183
#   macro avg       0.66      0.62      0.63      1183
#weighted avg       0.70      0.71      0.70      1183




