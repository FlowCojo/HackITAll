import random
import uuid
import joblib
import os
import numpy as np
from flask import request, jsonify
from flask.views import MethodView
from flask_smorest import Blueprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import csv
from schemas import WeaponSchema

blp = Blueprint("Game", __name__, description="Game AI endpoints")

# Full 60-word pool with type and cost
WORD_POOL = [
    {"word": "fire", "type": "strength", "cost": 5},
    {"word": "stone", "type": "strength", "cost": 6},
    {"word": "hammer", "type": "strength", "cost": 7},
    {"word": "blast", "type": "strength", "cost": 5},
    {"word": "quake", "type": "strength", "cost": 8},
    {"word": "iron", "type": "strength", "cost": 4},
    {"word": "storm", "type": "strength", "cost": 6},
    {"word": "fury", "type": "strength", "cost": 5},
    {"word": "rage", "type": "strength", "cost": 5},
    {"word": "shock", "type": "strength", "cost": 6},

    {"word": "wind", "type": "grace", "cost": 3},
    {"word": "feather", "type": "grace", "cost": 2},
    {"word": "petal", "type": "grace", "cost": 2},
    {"word": "stream", "type": "grace", "cost": 3},
    {"word": "dance", "type": "grace", "cost": 4},
    {"word": "whisper", "type": "grace", "cost": 3},
    {"word": "silk", "type": "grace", "cost": 2},
    {"word": "glide", "type": "grace", "cost": 3},
    {"word": "cloud", "type": "grace", "cost": 2},
    {"word": "aura", "type": "grace", "cost": 3},

    {"word": "logic", "type": "logic", "cost": 4},
    {"word": "reason", "type": "logic", "cost": 4},
    {"word": "code", "type": "logic", "cost": 5},
    {"word": "calculus", "type": "logic", "cost": 6},
    {"word": "theory", "type": "logic", "cost": 5},
    {"word": "axiom", "type": "logic", "cost": 4},
    {"word": "proof", "type": "logic", "cost": 4},
    {"word": "mind", "type": "logic", "cost": 4},
    {"word": "truth", "type": "logic", "cost": 3},
    {"word": "rule", "type": "logic", "cost": 3},

    {"word": "shadow", "type": "stealth", "cost": 6},
    {"word": "cloak", "type": "stealth", "cost": 5},
    {"word": "phantom", "type": "stealth", "cost": 6},
    {"word": "smoke", "type": "stealth", "cost": 5},
    {"word": "ghost", "type": "stealth", "cost": 5},
    {"word": "echo", "type": "stealth", "cost": 4},
    {"word": "blur", "type": "stealth", "cost": 3},
    {"word": "sneak", "type": "stealth", "cost": 4},
    {"word": "veil", "type": "stealth", "cost": 3},
    {"word": "drift", "type": "stealth", "cost": 4},

    {"word": "light", "type": "purity", "cost": 2},
    {"word": "halo", "type": "purity", "cost": 3},
    {"word": "shine", "type": "purity", "cost": 2},
    {"word": "blessing", "type": "purity", "cost": 3},
    {"word": "hope", "type": "purity", "cost": 2},
    {"word": "grace", "type": "purity", "cost": 2},
    {"word": "divine", "type": "purity", "cost": 4},
    {"word": "peace", "type": "purity", "cost": 2},
    {"word": "truth", "type": "purity", "cost": 3},
    {"word": "cleanse", "type": "purity", "cost": 3},
]

COUNTER_MAP = {
    "strength": "logic",
    "grace": "strength",
    "logic": "stealth",
    "stealth": "purity",
    "purity": "grace",
}

SESSION_SCORE = {
    "total_cost": 0,
    "rounds": 0
}

le = LabelEncoder()
types = [w["type"] for w in WORD_POOL]
le.fit(types)

MODEL_PATH = "ai_word_model_v2.pkl"

weapons = {}

def generate_training_data(n_samples=1000):
    X, y = [], []
    with open("training_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "system_word", "system_type",
            "counter_word", "counter_type",
            "cost", "penalty", "total_score", "label"
        ])

        for _ in range(n_samples):
            system_word = random.choice(WORD_POOL)
            counter_word = random.choice(WORD_POOL)

            system_type_encoded = le.transform([system_word["type"]])[0]
            counter_type_encoded = le.transform([counter_word["type"]])[0]
            cost = counter_word["cost"]
            penalty = random.choice([0, 1, 2])
            total_score = cost + penalty

            is_counter = COUNTER_MAP.get(system_word["type"]) == counter_word["type"]
            label = int(is_counter)

            X.append([system_type_encoded, counter_type_encoded, cost])
            y.append(label)

            writer.writerow([
                system_word["word"], system_word["type"],
                counter_word["word"], counter_word["type"],
                cost, penalty, total_score,
                label
            ])
    return np.array(X), np.array(y)

def train_and_generate_data_if_needed():
    global clf
    if not os.path.exists(MODEL_PATH):
        X, y = generate_training_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        joblib.dump(clf, MODEL_PATH)
    else:
        clf = joblib.load(MODEL_PATH)

train_and_generate_data_if_needed()

@blp.route("/get-challenge")
class GetChallenge(MethodView):
    @blp.response(200)
    def get(self):
        word = random.choice(WORD_POOL)
        return {"challenge_word": word}

@blp.route("/get-suggestion/<string:system_word>")
class GetSuggestion(MethodView):
    @blp.response(200)
    def get(self, system_word):
        system = next((w for w in WORD_POOL if w["word"] == system_word), None)
        if not system:
            return {"error": "System word not found in word pool."}, 404

        system_encoded = le.transform([system["type"]])[0]
        best_choice = None
        best_prob = -1

        for word in WORD_POOL:
            counter_encoded = le.transform([word["type"]])[0]
            features = np.array([[system_encoded, counter_encoded, word["cost"]]])
            prob = clf.predict_proba(features)[0][1]
            if prob > best_prob:
                best_prob = prob
                best_choice = word

        is_counter = COUNTER_MAP.get(system["type"]) == best_choice["type"]
        reason = "AI suggests this word as a strong counter" if is_counter else "AI suggests cheapest safe option"

        return {
            "suggestion": best_choice,
            "reason": reason,
            "confidence": round(best_prob, 3)
        }

@blp.route("/submit-word")
class SubmitWord(MethodView):
    @blp.arguments(WeaponSchema)
    @blp.response(200)
    def post(self, weapon_data):
        system_word = random.choice(WORD_POOL)
        is_counter = COUNTER_MAP.get(system_word["type"]) == weapon_data["type"]
        penalty = 0 if is_counter else random.choice([1, 2])
        total = weapon_data["cost"] + penalty

        SESSION_SCORE["total_cost"] += total
        SESSION_SCORE["rounds"] += 1

        weapon_id = uuid.uuid4().hex
        weapon = {**weapon_data, "id": weapon_id}
        weapons[weapon_id] = weapon

        return {
            "result": "accepted",
            "is_counter": is_counter,
            "penalty": penalty,
            "total": total,
            "session_score": SESSION_SCORE["total_cost"],
            "session_rounds": SESSION_SCORE["rounds"],
            "system_word": system_word,
            "chosen_word": weapon
        }
