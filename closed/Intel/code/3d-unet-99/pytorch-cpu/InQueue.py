import sys
import numpy as np
import logging

from InputItem import InputItem
from baseInQueue import baseInQueue

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("3DUnet-InQueue")

class InQueue(baseInQueue):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
