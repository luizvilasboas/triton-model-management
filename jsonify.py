from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson
import config_pb2
import json

def pbtxt_to_dict(pbtxt_file):
    message = config_pb2.ModelConfig()

    with open(pbtxt_file, 'r') as f:
        pbtxt_content = f.read()
        text_format.Parse(pbtxt_content, message)

    json_content = MessageToJson(
        message, including_default_value_fields=True, preserving_proto_field_name=True)

    json_content = json.loads(json_content)

    return json_content
