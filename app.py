from flask import Flask
from flask_smorest import Api
from resources.game import blp as GameBlueprint
from resources.weapon import blp as WeaponBlueprint  # import blueprint-ul pentru arme
from resources.game import train_and_generate_data_if_needed




app = Flask(__name__)

app.config["PROPAGATE_EXCEPTIONS"] = True
app.config["API_TITLE"] = "Words of Power API"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.3"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

train_and_generate_data_if_needed()  # rulăm înainte să pornească aplicația
api = Api(app)

# Înregistrare blueprint-uri
api.register_blueprint(GameBlueprint)
api.register_blueprint(WeaponBlueprint)  # adăugat pentru endpointurile weapon

