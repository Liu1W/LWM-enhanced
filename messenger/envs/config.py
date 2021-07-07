"""
Default config settings used in the envs. Changing the config settings here
will have a global effect on the entire messenger package.
"""

import itertools
from pathlib import Path
import json
from collections import namedtuple

"""
An entity namedtuple consists of the entity's name (e.g. alien) and the
id (symbol) of the entity used to represent the entity in the state
"""
Entity = namedtuple("Entity", "name id")

"""
A Game namedtuple specifies an assignment of three Entity namedtuples to the roles
enemy, message, goal.
"""
Game = namedtuple("Game", "enemy message goal")

"""
ACTIONS (user set):
    Maps actions to ints that is consistent with those used in PyVGDL
STATE_HEIGHT, STATE_WIDTH (user set):
    Dimensions of the game state. Note that these must be consistent with the level
    files that PyVGDL reads.
_all_npcs (user set):
    List of all the npcs (non-player characters) in the game. Aside from the obvious 
    effects of changing the ent