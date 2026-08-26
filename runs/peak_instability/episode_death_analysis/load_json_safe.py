import json

def load_gate_json(path):
    """Some gate JSON files have a stray [nes_core::Pool] log line prepended
    before the actual JSON body. Strip any leading non-'{' lines."""
    text = open(path).read()
    idx = text.find('{')
    return json.loads(text[idx:])
