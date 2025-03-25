import uuid
from flask.views import MethodView
from flask_smorest import Blueprint, abort
from schemas import WeaponSchema

blp = Blueprint("Weapons", __name__, description="Operations on weapons")

#  Mock db as a dictionary
weapons = {
    "1": {"id": "1", "name": "Katana", "cost": 15.99},
    "2": {"id": "2", "name": "Longbow", "cost": 10.50},
    "3": {"id": "3", "name": "Throwing Axe", "cost": 7.25}
}

@blp.route("/weapon/<string:weapon_id>")
class WeaponResource(MethodView):
    @blp.response(200, WeaponSchema)
    def get(self, weapon_id):
        try:
            return weapons[weapon_id]
        except KeyError:
            abort(404, message="Weapon not found.")

    def delete(self, weapon_id):
        try:
            del weapons[weapon_id]
            return {"message": "Weapon deleted."}
        except KeyError:
            abort(404, message="Weapon not found.")

@blp.route("/weapon")
class WeaponList(MethodView):
    @blp.response(200, WeaponSchema(many=True))
    def get(self):
        return list(weapons.values())

    @blp.arguments(WeaponSchema)
    @blp.response(201, WeaponSchema)
    def post(self, weapon_data):
        for weapon in weapons.values():
            if weapon_data["name"] == weapon["name"]:
                abort(400, message="Weapon already exists.")

        weapon_id = uuid.uuid4().hex
        weapon = {**weapon_data, "id": weapon_id}
        weapons[weapon_id] = weapon
        return weapon
