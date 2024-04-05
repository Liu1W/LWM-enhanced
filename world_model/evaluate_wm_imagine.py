
import os
import sys

sys.path.append("..")
import random
import json

from collections import defaultdict
import numpy as np
import pprint

from chatgpt_groundings.utils import load_gpt_groundings
from messenger.envs import make_env
from train_wm import make_dataset, make_model
from transformer_world_model.utils import ENTITY_IDS
import flags


ACTIONS = [(0, -1, 0), (1, 1, 0), (2, 0, -1), (3, 0, 1), (4, 0, 0)]


def get_entities(o):
    return o.reshape(-1, 4).max(0).tolist()


def get_distance(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def find_entity_coordinates(o, e=None):
    entities = get_entities(o)
    if e is None:
        e = 15 if 15 in entities else 16
    if e not in entities:
        return None, None
    c = entities.index(e)
    p = o.reshape(-1, 4)[:, c].tolist().index(e)
    x = p // 10
    y = p % 10
    return x, y


def compute_distances(observations, e):
    dists = []
    for o in observations:
        ex, ey = find_entity_coordinates(o, e)
        px, py = find_entity_coordinates(o)
        # player is dead, skip
        if px is None:
            continue
        # player is alive
        if ex is not None:
            d = get_distance(ex, ey, px, py)
        else:
            d = -1
        dists.append(d)
    return dists


def next_positon(x, y, a):
    nx = x + ACTIONS[a][1]
    ny = y + ACTIONS[a][2]
    nx = max(min(nx, 9), 0)
    ny = max(min(ny, 9), 0)
    return nx, ny


def get_role_by_id(manual, id):
    for e in manual:
        if ENTITY_IDS[e[0]] == id:
            return e[-1]

