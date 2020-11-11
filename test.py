aitext = open("C:\\Users\\gerhas\\Documents\\GitHub\\bert\\text\\AI.txt", "r").read()
shoppingtext = open("C:\\Users\\gerhas\\Documents\\GitHub\\bert\\text\\shopping.txt", "r").read()
query = open("C:\\Users\\gerhas\\Documents\\GitHub\\bert\\text\\query.txt", "r").read()

import scipy.spatial.distance

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

sentences = [aitext, 
             shoppingtext]
 

# Each sentence is encoded as a 1-D vector with 78 columns
sentence_embeddings = model.encode(sentences)

print('Sample BERT embedding vector - length', len(sentence_embeddings[0]))


# code adapted from https://github.com/UKPLab/sentence-transformers/blob/master/examples/application_semantic_search.py

#query = 'We are talking to Samsung' #@param {type: 'string'}

queries = [query]
query_embeddings = model.encode(queries)

# Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
number_top_matches = 3 #@param {type: "number"}

print("Semantic Search Results")

for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query[0:30])
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:number_top_matches]:
        print(sentences[idx].strip()[0:30], "(Cosine Score: %.4f)" % (1-distance))