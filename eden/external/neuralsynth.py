import math
import time
import sys

import copy
import json
import os
from os import listdir
from os.path import isfile, join
from random import random
from io import BytesIO
from enum import Enum
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
import numpy as np
import scipy.misc
import cv2
import tensorflow as tf


import eden.setup
from eden.canvas import *
from eden.utils import processing

is_setup = False

def setup():
    global sequence, bookmarks, lapnorm, is_setup
    neural_synth = eden.setup.get_external_repo_dir('neural-synth')
    sys.path.insert(0, neural_synth)
    import lapnorm, canvas, mask, util, bookmarks, generate, sequence
    if not is_setup:
        inception_graph = os.path.join(neural_synth, 'data/inception5h/tensorflow_inception_graph.pb')
        lapnorm.setup(inception_graph)
        is_setup = True
    

def run(channels, output, attributes, canvas, mask):
    seq = sequence.Sequence(attributes)
    seq.append(channels, mask, canvas, output['num_frames'], 0)
    seq.loop_to_beginning(output['num_loop_frames'])
    img = lapnorm.generate(seq, attributes, output, start_from=0, preview=False)
    return img


def get_random_favorites(layer_alias, n):
    layer = bookmarks.get_bookmarks_via_alias(layer_alias)
    return bookmarks.get_random_favorites(layer, n)