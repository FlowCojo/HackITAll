from marshmallow import Schema, fields, validate

class WeaponSchema(Schema):
    id = fields.Str(dump_only=True)  # ID-ul se generează automat, nu trebuie trimis de client
    name = fields.Str(
        required=True,
        validate=validate.Length(min=1),
        metadata={"description": "The name of the weapon"}
    )
    cost = fields.Float(
        required=True,
        validate=validate.Range(min=0.01),
        metadata={"description": "The cost of the weapon in USD"}
    )
    type = fields.Str(
        required=True,
        validate=validate.OneOf(["strength", "grace", "logic", "stealth", "purity"]),
        metadata={"description": "The category/type of the weapon"}
    )
