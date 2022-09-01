from flask import Flask
from sudoku_server.apps import solve

app = Flask(__name__, static_folder='sudoku_server/static')
app.debug = True
app.register_blueprint(solve.solve_blueprint)
print(app.url_map)

if __name__ == '__main__':
    app.run(host="0.0.0.0")
