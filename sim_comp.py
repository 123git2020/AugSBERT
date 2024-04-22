from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences1 = [
    "The cat sits outside",
    "A man is playing guitar",
    "The new movie is bad",
    "The teacher is reading the book"
]

sentences2 = [
    "The dog plays in the garden",
    "A woman is eating cookie",
    "The new movie is so great",
    "The teacher is drinking the tea"
]

# Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

# Compute cosine-similarities
cosine_scores = util.cos_sim(embeddings1, embeddings2)

# Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(
        sentences1[i], sentences2[i], cosine_scores[i][i]
    ))