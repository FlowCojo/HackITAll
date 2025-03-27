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
from sklearn.metrics import classification_report
import csv
from schemas import WeaponSchema

blp = Blueprint("Game", __name__, description="Game AI endpoints")

# Word pool used by the AI system
WORD_POOL = [
    {"word": "fire", "type": "strength", "cost": 5},
    {"word": "wind", "type": "grace", "cost": 3},
    {"word": "logic", "type": "logic", "cost": 4},
    {"word": "shadow", "type": "stealth", "cost": 6},
    {"word": "light", "type": "purity", "cost": 2},
]

le = LabelEncoder()
types = [w["type"] for w in WORD_POOL]
le.fit(types)

MODEL_PATH = "ai_word_model_v2.pkl"

# Store for game-generated weapons (shared between endpoints)
weapons = {}

# Train model if needed

def generate_training_data(n_samples=1000):
    X, y = [], []
    csv_file = "training_data.csv"

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "system_word", "system_type",
            "counter_word", "counter_type",
            "cost", "penalty", "total_score",
            "label"
        ])

        for _ in range(n_samples):
            system_word = random.choice(WORD_POOL)
            counter_word = random.choice(WORD_POOL)

            system_type_encoded = le.transform([system_word["type"]])[0]
            counter_type_encoded = le.transform([counter_word["type"]])[0]
            cost = counter_word["cost"]
            penalty = random.choice([0, 1, 2])
            total_score = cost + penalty

            label = int(counter_word["type"] != system_word["type"] and total_score <= 5)

            X.append([system_type_encoded, counter_type_encoded, cost])
            y.append(label)

            writer.writerow([
                system_word["word"], system_word["type"],
                counter_word["word"], counter_word["type"],
                cost, penalty, total_score,
                label
            ])

    print(f"✅ {n_samples} training samples saved to {csv_file}")
    return np.array(X), np.array(y)

if not os.path.exists(MODEL_PATH):
    X, y = generate_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, MODEL_PATH)
else:
    clf = joblib.load(MODEL_PATH)

def train_and_generate_data_if_needed():
    if not os.path.exists(MODEL_PATH):
        X, y = generate_training_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        joblib.dump(clf, MODEL_PATH)
        print("✅ Model trained and CSV data generated.")
    else:
        print("ℹ️ Model already exists. Skipping training.")

# Pick system word
@blp.route("/get-challenge")
class GetChallenge(MethodView):
    @blp.response(200)
    def get(self):
        word = random.choice(WORD_POOL)
        return {"challenge_word": word}

# Suggest a word based on AI prediction
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

        if best_prob > 0.9:
            return {
                "suggestion": best_choice,
                "reason": "AI suggests this word as a strong counter",
                "confidence": round(best_prob, 3)
            }
        else:
            min_cost = min(WORD_POOL, key=lambda x: x["cost"])
            return {
                "suggestion": min_cost,
                "reason": "No strong counter found. Suggesting lowest cost option.",
                "confidence": round(best_prob, 3)
            }

# Submit word and score it
@blp.route("/submit-word")
class SubmitWord(MethodView):
    @blp.arguments(WeaponSchema)
    @blp.response(200)
    def post(self, weapon_data):
        system_word = random.choice(WORD_POOL)  # Simulated challenge
        system_encoded = le.transform([system_word["type"]])[0]
        counter_encoded = le.transform([weapon_data["type"]])[0]
        features = np.array([[system_encoded, counter_encoded, weapon_data["cost"]]])

        prob = clf.predict_proba(features)[0][1]
        penalty = random.choice([0, 1, 2])
        total = weapon_data["cost"] + penalty

        weapon_id = uuid.uuid4().hex
        weapon = {
            **weapon_data,
            "id": weapon_id
        }
        weapons[weapon_id] = weapon

        return {
            "result": "accepted",
            "confidence": round(prob, 3),
            "penalty": penalty,
            "total": total,
            "system_word": system_word,
            "chosen_word": weapon
        }
