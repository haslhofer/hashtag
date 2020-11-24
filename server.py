from flask import Flask, jsonify, request
from http import HTTPStatus
from sentence_transformers import SentenceTransformer
import scipy.spatial.distance
import datetime
import spacy


app = Flask(__name__)

recipes = [
    {
        'id': 1,
        'name': 'Egg Salad',
        'description': 'This is a lovely egg salad recipe.'
    },
    {
        'id': 2, 'name': 'Tomato Pasta',
        'description': 'This is a lovely tomato pasta recipe.'
    }
]


@app.route('/ner', methods=['GET'])
def get_ner():
    query = open("text\\query.txt", "r").read()
    #sentence = "Gerald Haslhofer is the most recent author of the article about AI"
    doc = nlp(query)
    entities = []
    recognized = []

    for ent in doc.ents:
        if (ent.label_ == 'PERSON'):    
            if (not (recognized.__contains__(ent.text))):
                anEntity = [
                    {
                        'text' : ent.text,
                        'label' : ent.label_
                    }
                ]
                recognized.append(ent.text)
                entities.append(anEntity)
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    return jsonify({'Table1': entities})


@app.route('/recipes', methods=['GET'])
def get_recipes():

    confidences = []

    
    print ("Start:" + str(datetime.datetime.now()))

    sentences = []
    tags = []
    conf = open("C:\\Users\\gerhas\\Documents\\GitHub\\hashtag\\text\\config.txt", "r").readlines()
    a = 0
    confpaths = []
    for elem in conf:
        confpath = "text\\" + elem.strip() + ".txt"
        tags.append(elem.strip())
        conftext = open(confpath, "r").read()
        sentences.append(conftext)

    query = open("text\\query.txt", "r").read()

    # Each sentence is encoded as a 1-D vector with 78 columns
    sentence_embeddings = model.encode(sentences)

    queries = [query]
    query_embeddings = model.encode(queries)

    # Find the closest 3 sentences of the corpus for each query sentence based on cosine similarity
    number_top_matches = 3 

    
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        for idx, distance in results[0:number_top_matches]:
            aConfidence = [
                {
                    'hashtag': tags[idx],
                    'confidence': (1-distance)
                }
            ]
            confidences.append(aConfidence)


            print(tags[idx])
            print("%.4f" % (1-distance))
        
    print ("End:" + str(datetime.datetime.now()))
    return jsonify({'Table1': confidences})


@app.route('/recipes/<int:recipe_id>', methods=['GET'])
def get_recipe(recipe_id):
    recipe = next((recipe for recipe in recipes if recipe['id'] == recipe_id), None)

    if recipe:
        return jsonify(recipe)

    return jsonify({'message': 'recipe not found'}), HTTPStatus.NOT_FOUND


@app.route('/recipes', methods=['POST'])
def create_recipe():
    data = request.get_json()

    name = data.get('name')
    description = data.get('description')

    recipe = {
        'id': len(recipes) + 1,
        'name': name,
        'description': description
    }

    recipes.append(recipe)

    return jsonify(recipe), HTTPStatus.CREATED


@app.route('/recipes/<int:recipe_id>', methods=['PUT'])
def update_recipe(recipe_id):
    recipe = next((recipe for recipe in recipes if recipe['id'] == recipe_id), None)

    if not recipe:
        return jsonify({'message': 'recipe not found'}), HTTPStatus.NOT_FOUND

    data = request.get_json()

    recipe.update(
        {
            'name': data.get('name'),
            'description': data.get('description')
        }
    )

    return jsonify(recipe)

if __name__ == '__main__':
    
    nlp = spacy.load('en_core_web_sm')
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    app.run()