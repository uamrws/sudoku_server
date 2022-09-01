from flask import Blueprint

solve_blueprint = Blueprint("solve_route", __name__, template_folder='../../template')

from . import views
