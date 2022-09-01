from . import solve_blueprint
from flask import request, jsonify, render_template, render_template_string
from sudoku_server.utils.sudoku_solver import SudokuHandler
import time
import json


@solve_blueprint.route("/", methods=["GET"])
def index():
    return render_template("index.html", ran=int(time.time()))


@solve_blueprint.route("/solve", methods=["POST"])
def solve():
    rows = json.loads(request.form.get('rows'))
    is_all = json.loads(request.form.get('is_all'))
    nums = 0
    for row in rows:
        pass
    sudoku = SudokuHandler()
    sudoku.build_from_list(rows)
    if is_all:
        results = sudoku.query_all()
    else:
        results = sudoku.query_one()

    string = render_template('solve.html', results=results, ran=int(time.time()))
    return jsonify(string)


@solve_blueprint.route("/temp", methods=["GET"])
def temp():
    return render_template("temp.html", ran=int(time.time()))

@solve_blueprint.route("/temp2", methods=["POST"])
def temp1():
    haha = request.form.get('haha')
    print(haha)
    return "haha"