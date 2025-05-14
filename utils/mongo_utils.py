from typing import Optional, List, Text
from mongoengine.base import BaseDocument
import mongoengine as me
import bson


def mongo_to_dict(
    doc: BaseDocument, fields_to_ignore: Optional[List[Text]] = None
) -> dict:
    fields_to_ignore = fields_to_ignore if fields_to_ignore else []

    result = {}
    for field_name, field in doc._fields.items():
        if field_name in fields_to_ignore:
            continue
        value = getattr(doc, field_name)

        if isinstance(field, me.ReferenceField):
            result[field_name] = str(value.id) if value else None
        elif isinstance(value, BaseDocument):
            result[field_name] = mongo_to_dict(value, fields_to_ignore)
        elif isinstance(value, list):
            result[field_name] = [
                (
                    mongo_to_dict(item, fields_to_ignore)
                    if isinstance(item, BaseDocument)
                    else item
                )
                for item in value
            ]
        elif isinstance(value, dict):
            result[field_name] = {
                k: (
                    str(v.id)
                    if isinstance(field.field, me.ReferenceField) and v
                    else (
                        mongo_to_dict(v, fields_to_ignore)
                        if isinstance(v, BaseDocument)
                        else v
                    )
                )
                for k, v in value.items()
            }
        elif isinstance(value, bson.ObjectId):
            result[field_name] = str(value)
        else:
            result[field_name] = value

    return result
