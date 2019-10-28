import logging
from search import Searcher
from flask import Flask, request, render_template

logging.basicConfig(filename="runtime.log", level=logging.INFO)
root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                              datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
root.addHandler(handler)

searcher = Searcher()
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("main.html")

@app.route('/search', methods=['POST'])
def query():
    method = request.form.get("method")
    query = request.form.get("query")
    matches = searcher[method].search(query)
    return render_template("main.html", matches=matches)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)