from enum import Enum

SUBFRAMES_IN_FRAME = 10
HALF = 0.5


class Phase(Enum):
    TRAIN = 'train'
    TEST = 'test'
