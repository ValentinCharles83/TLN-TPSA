import matplotlib.pyplot as plt

def visualize_sentiment(data_paths):
    pos_counts = []
    neg_counts = []

    for path in data_paths:
        with open(path, 'r', encoding='utf-8') as f:
            pos = 0
            neg = 0
            for line in f:
                _, _, pos_score, neg_score = eval(line)
                if pos_score > 0:
                    pos += 1
                if neg_score > 0:
                    neg += 1
            pos_counts.append(pos)
            neg_counts.append(neg)

    # Assuming you have the same order of files for pos_counts and neg_counts
    labels = ['Train 1', 'Train 2', 'Test Gold 1', 'Test Gold 2']
    x = range(len(labels))

    plt.bar(x, pos_counts, width=0.4, label='Positive', color='green', align='center')
    plt.bar(x, neg_counts, width=0.4, label='Negative', color='red', align='edge')
    plt.xlabel('Files')
    plt.ylabel('Counts')
    plt.title('Sentiment Distribution Across Files')
    plt.xticks(x, labels, rotation='vertical')
    plt.legend()
    plt.show()

# Call visualize_sentiment with paths to your processed files
visualize_sentiment(['Restaurants_Train_Sentiment_Analysis_Processed.txt', 'Laptop_Train_Sentiment_Analysis_Processed.txt', 'Restaurants_Test_Gold_Sentiment_Analysis_Processed.txt', 'Laptop_Test_Gold_Sentiment_Analysis_Processed.txt'])
