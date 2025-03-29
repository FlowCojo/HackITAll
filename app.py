from flask import Flask
from flask_smorest import Api
from flask_cors import CORS
from resources import game
from resources.weapon import blp as WeaponBlueprint

app = Flask(__name__)
CORS(app)

app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["API_TITLE"] = "Words of Power API"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.3"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

api = Api(app)
api.register_blueprint(game.blp)
api.register_blueprint(WeaponBlueprint)

@app.route("/session-score")
def get_session_score():
    return {
        "total_cost": game.SESSION_SCORE["total_cost"],
        "rounds": game.SESSION_SCORE["rounds"]
    }

@app.route("/reset-session", methods=["POST"])
def reset_session_score():
    game.SESSION_SCORE["total_cost"] = 0
    game.SESSION_SCORE["rounds"] = 0
    return {"message": "Session score reset."}

if __name__ == "__main__":
    game.train_and_generate_data_if_needed()
    app.run(debug=True)
