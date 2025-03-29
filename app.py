from flask import Flask, request
from flask_smorest import Api
from resources.weapon import blp as ItemBlueprint

app = Flask(__name__)
CORS(app)

app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["API_TITLE"] = "Weapons REST API"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.3"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

api = Api(app)
api.register_blueprint(game.blp)
api.register_blueprint(WeaponBlueprint)

api.register_blueprint(ItemBlueprint)
