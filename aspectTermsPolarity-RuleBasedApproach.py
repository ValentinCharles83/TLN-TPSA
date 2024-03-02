import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import sentiwordnet as swn

nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('punkt')

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []

    for sentence in root.findall('.//sentence'):
        text = sentence.find('text').text
        aspect_terms_element = sentence.find('aspectTerms')
        aspect_categories_element = sentence.find('aspectCategories')
        
        if aspect_terms_element is not None:
            aspect_terms = [
                (at.get('term'), at.get('polarity'), int(at.get('from')), int(at.get('to')))
                for at in aspect_terms_element.findall('aspectTerm')
            ]
        else:
            aspect_terms = []
        
        if aspect_categories_element is not None:
            aspect_categories = [
                (ac.get('category'), ac.get('polarity'))
                for ac in aspect_categories_element.findall('aspectCategory')
            ]
        else:
            aspect_categories = []
        
        data.append((text, aspect_terms, aspect_categories))
    
    return data

def adjust_sentiment_score_based_on_pos(word, pos_tag, sentiment_score):
    """
    Adjusts the sentiment score of a word based on its part-of-speech tag.
    """
    if pos_tag.startswith('JJ'):  # Adjective
        return sentiment_score * 1.5
    elif pos_tag.startswith('RB'):  # Adverb
        return sentiment_score * 1.2
    elif pos_tag.startswith('NN') or pos_tag.startswith('VB'):  # Noun or Verb
        return sentiment_score
    else:
        return sentiment_score * 0.8  # Other parts of speech have less impact

def calculate_sentiment(aspect, sentence, window_size=3):
    words = nltk.word_tokenize(sentence)
    aspect_tokens = nltk.word_tokenize(aspect)
    pos_tags = nltk.pos_tag(words)

    aspect_index = None
    for i in range(len(words) - len(aspect_tokens) + 1):
        if words[i:i+len(aspect_tokens)] == aspect_tokens:
            aspect_index = i
            break

    if aspect_index is None:
        return 0

    start_index = max(0, aspect_index - window_size)
    end_index = min(len(words), aspect_index + len(aspect_tokens) + window_size)
    sentiment_score = 0

    for i in range(start_index, end_index):
        word = words[i]
        pos_tag = pos_tags[i][1]
        synsets = list(swn.senti_synsets(word))
        if synsets:
            base_score = synsets[0].pos_score() - synsets[0].neg_score()
            adjusted_score = adjust_sentiment_score_based_on_pos(word, pos_tag, base_score)
            sentiment_score += adjusted_score

    return sentiment_score

def determine_polarity(sentiment_score):
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'

def evaluate_predictions(predictions, gold_standard):
    # Assume predictions and gold_standard are dictionaries with aspect terms as keys
    # and sentiment polarities ('positive', 'negative', 'neutral') as values.
    true_positives = {k: 0 for k in predictions.keys()}
    false_positives = {k: 0 for k in predictions.keys()}
    false_negatives = {k: 0 for k in predictions.keys()}
    
    for aspect, polarity in predictions.items():
        if aspect in gold_standard and polarity == gold_standard[aspect]:
            true_positives[aspect] += 1
        elif aspect not in gold_standard or polarity != gold_standard[aspect]:
            false_positives[aspect] += 1
        if aspect in gold_standard and polarity != gold_standard[aspect]:
            false_negatives[aspect] += 1
    
    accuracy = sum(true_positives.values()) / len(predictions)
    recall = sum(true_positives.values()) / (sum(true_positives.values()) + sum(false_negatives.values()))
    F_measure = 2 * (accuracy * recall) / (accuracy + recall)
    
    return accuracy, recall, F_measure

def process_files(train_files, test_files_no_labels, test_files_gold):
    # Predictions is a dict with key: (aspect, sentence_id) to handle multiple aspects and sentences
    predictions = {}
    sentence_id = 0  # Initialize sentence_id if you need to uniquely identify sentences

    # Process test files without labels to predict sentiment
    for file_path in test_files_no_labels:
        data = parse_xml(file_path)
        for sentence_data in data:  # Iterate directly through the list
            sentence, aspects, _ = sentence_data  # Assuming you don't need aspect categories here
            for aspect_term, polarity, _, _ in aspects:  # Unpack aspect term details
                sentiment_score = calculate_sentiment(aspect_term, sentence)
                predictions[(aspect_term, sentence_id)] = determine_polarity(sentiment_score)
            sentence_id += 1  # Increment sentence_id for unique identification

    # Load gold standard for evaluation
    gold_standard = {}
    sentence_id = 0  # Reset or reuse sentence_id for consistency
    for file_path in test_files_gold:
        gold_data = parse_xml(file_path)
        for sentence_data in gold_data:
            sentence, aspects, _ = sentence_data
            for aspect_term, polarity, _, _ in aspects:
                gold_standard[(aspect_term, sentence_id)] = polarity  # Assuming you're using the gold standard polarity directly
            sentence_id += 1

    # Evaluate predictions against the gold standard
    accuracy, recall, F_measure = evaluate_predictions(predictions, gold_standard)

    # Print the evaluation results
    print(f'Accuracy: {accuracy:.4f}\nRecall: {recall:.4f}\nF-measure: {F_measure:.4f}')


# Example usage:
train_files = ["SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train.xml", "SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train.xml"]
test_files_no_labels = ["SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Test_NoLabels.xml", "SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Test_NoLabels.xml"]
test_files_gold = ["SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Test_Gold.xml", "SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Test_Gold.xml"]
process_files(train_files, test_files_no_labels, test_files_gold)