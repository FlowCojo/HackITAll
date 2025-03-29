import requests
from time import sleep
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 5

word = {
    1: {"text": "Feather", "cost": 1},
    2: {"text": "Coal", "cost": 1},
    3: {"text": "Pebble", "cost": 1},
    4: {"text": "Leaf", "cost": 2},
    5: {"text": "Paper", "cost": 2},
    6: {"text": "Rock", "cost": 2},
    7: {"text": "Water", "cost": 3},
    8: {"text": "Twig", "cost": 3},
    9: {"text": "Sword", "cost": 4},
    10: {"text": "Shield", "cost": 4},
    11: {"text": "Gun", "cost": 5},
    12: {"text": "Flame", "cost": 5},
    13: {"text": "Rope", "cost": 5},
    14: {"text": "Disease", "cost": 6},
    15: {"text": "Cure", "cost": 6},
    16: {"text": "Bacteria", "cost": 6},
    17: {"text": "Shadow", "cost": 7},
    18: {"text": "Light", "cost": 7},
    19: {"text": "Virus", "cost": 7},
    20: {"text": "Sound", "cost": 8},
    21: {"text": "Time", "cost": 8},
    22: {"text": "Fate", "cost": 8},
    23: {"text": "Earthquake", "cost": 9},
    24: {"text": "Storm", "cost": 9},
    25: {"text": "Vaccine", "cost": 9},
    26: {"text": "Logic", "cost": 10},
    27: {"text": "Gravity", "cost": 10},
    28: {"text": "Robots", "cost": 10},
    29: {"text": "Stone", "cost": 11},
    30: {"text": "Echo", "cost": 11},
    31: {"text": "Thunder", "cost": 12},
    32: {"text": "Karma", "cost": 12},
    33: {"text": "Wind", "cost": 13},
    34: {"text": "Ice", "cost": 13},
    35: {"text": "Sandstorm", "cost": 13},
    36: {"text": "Laser", "cost": 14},
    37: {"text": "Magma", "cost": 14},
    38: {"text": "Peace", "cost": 14},
    39: {"text": "Explosion", "cost": 15},
    40: {"text": "War", "cost": 15},
    41: {"text": "Enlightenment", "cost": 15},
    42: {"text": "Nuclear Bomb", "cost": 16},
    43: {"text": "Volcano", "cost": 16},
    44: {"text": "Whale", "cost": 17},
    45: {"text": "Earth", "cost": 17},
    46: {"text": "Moon", "cost": 17},
    47: {"text": "Star", "cost": 18},
    48: {"text": "Tsunami", "cost": 18},
    49: {"text": "Supernova", "cost": 19},
    50: {"text": "Antimatter", "cost": 19},
    51: {"text": "Plague", "cost": 20},
    52: {"text": "Rebirth", "cost": 20},
    53: {"text": "Tectonic Shift", "cost": 21},
    54: {"text": "Gamma-Ray Burst", "cost": 22},
    55: {"text": "Human Spirit", "cost": 23},
    56: {"text": "Apocalyptic Meteor", "cost": 24},
    57: {"text": "Earth’s Core", "cost": 25},
    58: {"text": "Neutron Star", "cost": 26},
    59: {"text": "Supermassive Black Hole", "cost": 35},
    60: {"text": "Entropy", "cost": 45},
}

model = SentenceTransformer('all-MiniLM-L6-v2')
word_texts = [info["text"] for info in word.values()]
word_ids = list(word.keys())
word_costs = [word[i]["cost"] for i in word_ids]
word_embeddings = model.encode(word_texts)

def what_beats(system_word_text, similarity_threshold=0.25, cost_penalty_weight=0.005):
    system_embedding = model.encode([system_word_text])[0]
    sims = cosine_similarity([system_embedding], word_embeddings)[0]

    scored_candidates = []
    for i, sim in enumerate(sims):
        cost = word_costs[i]
        score = sim - (cost_penalty_weight * cost)
        if sim >= similarity_threshold:
            scored_candidates.append((i, score, cost))

    if not scored_candidates:

        fallback_idx = sorted(
            range(len(word_embeddings)),
            key=lambda i: (word_costs[i], -sims[i])
        )[0]
        return word_ids[fallback_idx]

    best = max(scored_candidates, key=lambda x: x[1])
    return word_ids[best[0]]

def play_game(player_id):

    for round_id in range(1, NUM_ROUNDS+1):
        round_num = -1
        while round_num != round_id:
            response = requests.get(get_url)
            print(response.json())
            sys_word = response.json()['word']
            round_num = response.json()['round']

            sleep(1)

        if round_id > 1:
            status = requests.get(status_url)
            print(status.json())

        choosen_word = what_beats(sys_word)
        data = {"player_id": player_id, "word_id": choosen_word, "round_id": round_id}
        response = requests.post(post_url, json=data)
        print(response.json())

play_game("QvYX9qcMPH")