import traceback
from flask import jsonify, request, abort

from . import app
from sherlock.analytics.predictions import Predictions

predict = Predictions()


@app.route("/health")
def get_health():
    return "Health OK"


@app.route("/tags/nextTag", methods=["POST"])
def get_tag():
    try:
        input_tags = request.get_json()['tags']
        response = {}
        response['predicted_tags'] = predict.get_tag_prediction(tags=input_tags)
        return jsonify(data=response, error=False)

    except Exception:
        print(traceback.format_exc())
        abort(500)


@app.route("/search", methods=["POST"])
def get_relevant_products():
    try:
        input_query = request.get_json()['search_query']
        response = {}
        response['relevant_tags'] = predict.get_search_query_product(search_query=input_query)
        return str(response)

    except Exception:
        print(traceback.format_exc())
        abort(500)




