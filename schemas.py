from marshmallow import Schema, fields, validate

class WeaponSchema(Schema):
    id = fields.Str(dump_only=True)  # ID-ul se generează automat, nu trebuie trimis de client
    name = fields.Str(
        required=True,
        validate=validate.Length(min=1),
        description="The name of the weapon"
    )
    cost = fields.Float(
        required=True,
        validate=validate.Range(min=0.01),
        description="The cost of the weapon in USD"
    )
