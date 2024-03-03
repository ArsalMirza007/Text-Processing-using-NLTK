from sklearn.metrics import precision_score, recall_score

# Replace these with your actual true labels and predicted labels
true_labels = ['pos', 'neg', 'pos', 'neg', 'pos', 'neg']  # Example true labels
predicted_labels = ['pos', 'neg', 'pos', 'pos', 'pos', 'neg']  # Example predicted labels

# Calculate precision and recall for the 'pos' class
precision_pos = precision_score(true_labels, predicted_labels, pos_label='pos')
recall_pos = recall_score(true_labels, predicted_labels, pos_label='pos')

# Calculate precision and recall for the 'neg' class
precision_neg = precision_score(true_labels, predicted_labels, pos_label='neg')
recall_neg = recall_score(true_labels, predicted_labels, pos_label='neg')

# Display precision and recall values
print("Precision (pos):", precision_pos)
print("Recall (pos):", recall_pos)

print("Precision (neg):", precision_neg)
print("Recall (neg):", recall_neg)
