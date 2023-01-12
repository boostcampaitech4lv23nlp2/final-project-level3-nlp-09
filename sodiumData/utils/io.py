import orjson as json

from nuviAPI.hub import HubAPI

from typing import Union
from pathlib import Path


def read_json(path: Union[str, Path]) -> dict:
    """Read path with json library.
    """
    if isinstance(path, Path):
        path = path.as_posix()

    return json.loads(open(path, 'rb').read())


# lazy singleton
class SingleHubAPI:
    __hub = None

    @classmethod
    def __get_api(cls):
        return cls.__hub

    @classmethod
    def get_api(cls):
        cls.__hub = HubAPI()
        cls.get_api = cls.__get_api
        return cls.__hub
