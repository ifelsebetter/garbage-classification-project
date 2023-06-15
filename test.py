import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model("garbage_classification.h5")

test_data = "dataset/test"

input_size = (224, 224)
batch_size = 32
num_classes = 4

test_data_generator = ImageDataGenerator(rescale=1./255)
test_generator = test_data_generator.flow_from_directory(
    test_data,
    target_size=input_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

predictions = model.predict(test_generator)
predict_labels = np.argmax(predictions, axis=1)
true_labels = test_generator.classes


class_labels = ['glass', 'organic_waste', 'plastic']

# Initialize counters for each class within each folder
class_count = {0: 0, 1: 0, 2: 0}

# Iterate over the test samples and count the occurrences of each class within each folder
for i in range(len(test_generator.filenames)):
    filename = test_generator.filenames[i]
    folder_name = filename.split("/")[0]
    class_label = predict_labels[i]
    
    # Check if the class label is within the desired range
    if class_label >= 0 and class_label < len(class_labels):
        class_count[class_label] += 1
        print(f"Test image: {filename}, Predicted class: {class_labels[class_label]}")
    else:
        print(f"Invalid class label {class_label}.")

# Calculate the percentages for each class within each folder
folder_counts = {}
for class_label, count in class_count.items():
    # Check if the class label is within the desired range
    if class_label >= 0 and class_label < len(class_labels):
        folder_name = class_labels[class_label]
        if folder_name not in folder_counts:
            folder_counts[folder_name] = {}
        total_samples = np.sum(list(class_count.values()))
        percentage = (count / total_samples) * 100
        folder_counts[folder_name][class_label] = percentage

# Print the percentages for each class within each folder
for folder_name, class_percentages in folder_counts.items():
    print(f"\nFolder: {folder_name}")
    for class_label, percentage in class_percentages.items():
        print(f"Class {class_label}: {percentage:.2f}%")

accuracy = accuracy_score(true_labels, predict_labels)
print(f"\nTest Accuracy: {accuracy:.4f}")