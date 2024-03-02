
#import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')

import nltk
from xml.etree import ElementTree as ET

# Load the XML data from a file
def load_xml_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to process each sentence
def process_sentence(sentence):
    text = sentence.find('text').text
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    # For NER, using nltk's ne_chunk
    ne_tree = nltk.ne_chunk(pos_tags)
    
    return pos_tags, ne_tree

def main():
    # Path to XML data file
    file_path = 'SemEval\'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train.xml'
    output_path = 'Pre-Processed/Laptop_Train_Pre_Processed.txt'
    
    # Load and parse the XML data
    xml_data = load_xml_data(file_path)
    root = ET.fromstring(xml_data)
    
    with open(output_path, 'w', encoding='utf-8') as out_file:  # Open the output file
        # Process each sentence in the XML
        for sentence in root.findall('sentence'):
            pos_tags, ne_tree = process_sentence(sentence)
            
            # Write the processed data to the output file
            out_file.write("POS Tags: " + str(pos_tags) + "\n")
            out_file.write("Named Entities: " + str(ne_tree) + "\n\n")

if __name__ == "__main__":
    main()
