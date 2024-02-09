import pandas as pd

# The dictionary is now populated with each method as a key and a list of details as values
import pandas as pd
import tensorflow_hub as hub
import numpy as np
from scipy.spatial.distance import cosine
import tensorflow_hub as hub
import numpy as np
from scipy.spatial.distance import cosine

# Load the Universal Sentence Encoder model
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load the Excel file
file_path = 'Book.xlsx'  # Update this to the path where your Excel file is located
df = pd.read_excel(file_path)

# Initialize an empty dictionary to store methods and their corresponding details
methods_dict = {}

for index, row in df.iterrows():
    method = row['method'].strip()  # Remove any leading/trailing whitespace from method name
    detail = row['details'].strip()  # Remove any leading/trailing whitespace from detail
    
    # Check if the method is already a key in the dictionary
    if method in methods_dict:
        # If the method is already a key, append the new detail to its list
        methods_dict[method].append(detail)
    else:
        # If the method is not a key, create a new entry with the method as the key and a new list containing the detail
        methods_dict[method] = [detail]


# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# Function to compute embeddings for a list of ideas
def compute_embeddings(ideas):
    return model(ideas).numpy()

# Function to compute average cosine similarity within a group
def average_similarity_within_group(group_embeddings):
    num_ideas = len(group_embeddings)
    similarity_sum = 0
    count = 0
    for i in range(num_ideas):
        for j in range(i + 1, num_ideas):
            similarity_sum += cosine_similarity(group_embeddings[i], group_embeddings[j])
            count += 1
    return similarity_sum / count if count else 0

# Function to compute average cosine similarity between groups
def average_similarity_between_groups(group1_embeddings, group2_embeddings):
    similarity_sum = 0
    count = 0
    for vec1 in group1_embeddings:
        for vec2 in group2_embeddings:
            similarity_sum += cosine_similarity(vec1, vec2)
            count += 1
    return similarity_sum / count if count else 0

# Your provided dataset in a dictionary format
data = methods_dict

# Calculate and print average similarities within groups
for method, ideas in data.items():
    embeddings = compute_embeddings(ideas)
    avg_similarity = average_similarity_within_group(embeddings)
    print(f"Average similarity within {method}: {avg_similarity}")

# Calculate and print average similarities between groups
methods = list(data.keys())
for i in range(len(methods)):
    for j in range(i + 1, len(methods)):
        group1_embeddings = compute_embeddings(data[methods[i]])
        group2_embeddings = compute_embeddings(data[methods[j]])
        avg_similarity = average_similarity_between_groups(group1_embeddings, group2_embeddings)
        print(f"Average similarity between {methods[i]} and {methods[j]}: {avg_similarity}")
