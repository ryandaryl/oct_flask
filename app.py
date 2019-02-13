import os, json, hashlib
from io import BytesIO
import requests
from shapely.geometry import Polygon, mapping
from PIL import Image
from flask import Flask, jsonify, request
app = Flask(__name__)

def boundary_to_polygon(boundary):
  boundary = boundary + [(p[0] - 1, p[1]) for p in reversed(boundary)]
  polygon = mapping(Polygon(boundary).simplify(1))['coordinates'][0][:-1]
  return polygon

with open('boundaries_by_hash.json') as fh:
  for i in fh:
    boundaries_by_hash = json.loads(i)

@app.route("/predict_single_image", methods=['POST'])
def inference():
  url = request.get_json()['image_url']
  response = requests.get(url)
  image = Image.open(BytesIO(response.content))
  hash = hashlib.md5(image.tobytes()).hexdigest()
  boundary_names = ['ILM (ILM)', 'Inner boundary of RPE (IB_RPE)']
  ilm, inner_rpe = [boundary_to_polygon(boundaries_by_hash[hash][i]) for i in boundary_names]
  predictions = [
    {
      "class": "ILM",
      "confidence": 1,
      "polygon": ilm
    },
    {
      "class": "Inner RPE",
      "confidence": 1,
      "polygon": inner_rpe
    }
  ]
  return jsonify(predictions)

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host='0.0.0.0', port=port)
