
from js import document
import pyscript
from pyscript import Element

import numpy as np
from copy import deepcopy

from guess_who_dt.decision_tree import build_tree, print_tree
from guess_who_dt.utils import read_csv_file

CASTLE_CRASHERS_CSV = read_csv_file("./assets/data/CastleCrashers20Q.csv")
CASTLE_CRASHERS_CSV_HEADERS = CASTLE_CRASHERS_CSV[0]
CASTLE_CRASHERS_CHARACTERS = [ sub_list[0] for sub_list in np.array(CASTLE_CRASHERS_CSV)[:, 0].reshape(-1, 1)] 
CASTLE_CRASHERS_VALUES = np.array(CASTLE_CRASHERS_CSV[1:])[:, 1:].astype(int)

CASTLE_CRASHERS_DECISION_TREE = build_tree(
    dataset = CASTLE_CRASHERS_VALUES,
    depth = 0,
    min_split_length = 2,
    max_depth = 10
)

QUESTION_ID = "question"
CHARACTER_NAME_ID = "character_name"
CHARACTER_TRAIT_ID = "character_traits"
CHARACTER_REVEAL_ID = "character_reveal"

active_tree_node = deepcopy(CASTLE_CRASHERS_DECISION_TREE)

document.getElementById(QUESTION_ID).innerHTML = CASTLE_CRASHERS_CSV_HEADERS[active_tree_node.feature_value_index + 1]

def set_character_info(character_id:str):
    document.getElementById(CHARACTER_NAME_ID).innerHTML = "Character Name: " + CASTLE_CRASHERS_CHARACTERS[character_id]
    document.getElementById(CHARACTER_TRAIT_ID).innerHTML = "".join([f"<li>{CASTLE_CRASHERS_CSV_HEADERS[trait_index + 1]}</li>" for trait_index, trait_id in enumerate(CASTLE_CRASHERS_VALUES[character_id - 1]) if trait_id == 1 ])
def reset_tree():
    global active_tree_node
    active_tree_node = deepcopy(CASTLE_CRASHERS_DECISION_TREE)
    document.getElementById(QUESTION_ID).innerHTML = CASTLE_CRASHERS_CSV_HEADERS[active_tree_node.feature_value_index + 1]
    document.getElementById(CHARACTER_REVEAL_ID).innerHTML = ""

def navigate_tree_right():
    global active_tree_node

    # if there are no more features to the tree
    if not active_tree_node.feature_value_threshold_lte:
        return
    
    active_tree_node = active_tree_node.right_node
    
    if active_tree_node.leaf_value is not None:
        document.getElementById(QUESTION_ID).innerHTML = CASTLE_CRASHERS_CHARACTERS[int(active_tree_node.leaf_value)]
        document.getElementById(CHARACTER_REVEAL_ID).innerHTML = f'<image src="./assets/images/{CASTLE_CRASHERS_CHARACTERS[int(active_tree_node.leaf_value)].replace(" ","_")}_Portrait.webp" width="25%" class="character-portrait">'
    else:
        document.getElementById(QUESTION_ID).innerHTML = CASTLE_CRASHERS_CSV_HEADERS[active_tree_node.feature_value_index + 1]

def navigate_tree_left():
    global active_tree_node

    # if there are no more features to the tree
    if not active_tree_node.feature_value_threshold_lte:
        return
    
    active_tree_node = active_tree_node.left_node
    
    if active_tree_node.leaf_value is not None:
        document.getElementById(QUESTION_ID).innerHTML = CASTLE_CRASHERS_CHARACTERS[int(active_tree_node.leaf_value)]
        document.getElementById(CHARACTER_REVEAL_ID).innerHTML = f'<image src="./assets/images/{CASTLE_CRASHERS_CHARACTERS[int(active_tree_node.leaf_value)].replace(" ","_")}_Portrait.webp" width="25%" class="character-portrait">'
    else:
        document.getElementById(QUESTION_ID).innerHTML = CASTLE_CRASHERS_CSV_HEADERS[active_tree_node.feature_value_index + 1]