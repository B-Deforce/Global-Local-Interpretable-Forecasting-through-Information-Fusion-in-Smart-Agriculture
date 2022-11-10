"""Config class"""

import json

class Config:
    """
    Config class containing data, train, model, and sweep hyperparams
    """

    def __init__(self, data, model):
        self.data = data
        self.model = model

    @classmethod
    def from_json(cls, cfg):
        """
        Creates config form json
        """
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.model)


class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)
