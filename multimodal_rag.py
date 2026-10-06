import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bedrock_client import invoke_claude

with open('multimodal_knowledge_base.json', 'r') as f:
    knowledge_base = json.load(f)

for entry in knowledge_base[:5]:
    print(f"Text: {entry['caption_reference_description']}\nImage URL: {entry['image_url']}\n")

print("Total entries:", len(knowledge_base))

text_data = [entry['caption_reference_description'] for entry in knowledge_base]

vectorizer = TfidfVectorizer()
text_vectors = vectorizer.fit_transform(text_data)

print("Text Embeddings Shape:", text_vectors.shape)


def retrieve(query):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, text_vectors)
    most_relevant_idx = similarities.argmax()
    return knowledge_base[most_relevant_idx], similarities[0, most_relevant_idx]


query = "Where are the Rocky Mountains?"
most_relevant_entry, score = retrieve(query)

print("\nQuery:", query)
print("Most Relevant Entry:")
print(f"Text: {most_relevant_entry['caption_reference_description']}\nImage URL: {most_relevant_entry['image_url']}")
print(f"Similarity score: {score:.4f}")

for test_query in ["cereals", "space exploration", "quarterly earnings guidance for semiconductor stocks"]:
    entry, test_score = retrieve(test_query)
    print(f"\nQuery: {test_query}")
    print(f"Text: {entry['caption_reference_description']}")
    print(f"Image URL: {entry['image_url']}")
    print(f"Similarity score: {test_score:.4f}")

prompt = (
    "You are an expert assistant that can answer questions based on given information.\n\n"
    f"Using the following information, answer the query: {query}\n\n"
    f"Text: {most_relevant_entry['caption_reference_description']}\n"
)

result = invoke_claude(prompt, max_tokens=2000)
print("\nGenerated Answer:", result["text"])


def answer_query(query):
    most_relevant_entry, score = retrieve(query)

    print("\nRetrieved Context:")
    print(f"Text: {most_relevant_entry['caption_reference_description']}")
    print(f"Image URL: {most_relevant_entry['image_url']}")
    print(f"Similarity score: {score:.4f}\n")

    prompt = (
        "You are an expert assistant that can answer questions based on given information.\n\n"
        f"Using the following information, answer the query: {query}\n\n"
        f"Text: {most_relevant_entry['caption_reference_description']}\n"
    )

    result = invoke_claude(prompt, max_tokens=2000)

    print("Generated Answer:", result["text"])


answer_query("What are some types of grains?")