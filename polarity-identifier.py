import nltk
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('sentiwordnet')

def process_file(file_path, output_path):
    root = ET.parse(file_path).getroot()
    stop_words = set(stopwords.words('english'))
    sentiment_data = []  # This will store tuples of (word, POS, positive_score, negative_score)

    for sentence in root.findall('.//sentence'):
        text = sentence.find('text').text
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)

        for word, tag in pos_tags:
            if word.lower() in stop_words:
                continue  # Skip stop words

            # Convert the POS tag to a format used by SentiWordNet
            wn_tag = convert_to_wn_tag(tag)
            if wn_tag is None:
                continue  # If there's no equivalent SentiWordNet tag, skip the word

            synsets = list(swn.senti_synsets(word, wn_tag))
            if not synsets:
                continue  # If the word is not in SentiWordNet, skip it

            # For simplicity, just take the first synset
            synset = synsets[0]
            sentiment_data.append((word, tag, synset.pos_score(), synset.neg_score()))

    with open(output_path, 'w', encoding='utf-8') as f:
        for data in sentiment_data:
            f.write(f"{data}\n")

def convert_to_wn_tag(tag):
    if tag.startswith('N'):
        return 'n'  # Noun
    if tag.startswith('V'):
        return 'v'  # Verb
    if tag.startswith('J'):
        return 'a'  # Adjective
    if tag.startswith('R'):
        return 'r'  # Adverb
    return None

process_file('SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train.xml', 'Sentiment_Analysis_Processed/Laptop_Train_Sentiment_Analysis_Processed.txt')
