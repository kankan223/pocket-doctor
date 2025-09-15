import os
import json
import uuid
import math
import re
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, flash
from werkzeug.utils import secure_filename
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "gif"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "change_this_for_prod"  # change in production

# Load mapping
with open(os.path.join(BASE_DIR, "mapping.json"), "r", encoding="utf-8") as f:
    KB = json.load(f)

SESSIONS = {}  # simple in-memory session store {session_id: result_obj}


# configuration: tune these
WEIGHT_BASE = 0.5         # baseline for every condition
WEIGHT_REQUIRED = 1.5     # per required symptom match
WEIGHT_SUPPORTING = 0.8   # per supporting symptom match
WEIGHT_RED = 2.5          # per red-flag match (high)
PENALTY_MISSING_REQUIRED = -0.5  # penalty if a required symptom explicitly absent? (optional)



def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def extract_keywords(text, checked_list=None, kb=KB):
    """
    Returns a set of normalized keywords found in text + checked_list.
    - text: raw user text
    - checked_list: list from form checkboxes (strings)
    - kb: knowledge base loaded from mapping.json (contains synonyms and conditions)
    """
    if not text:
        text = ""
    text_norm = text.lower()
    # remove extra punctuation but keep internal hyphens
    text_norm = re.sub(r"[^\w\s\-']", " ", text_norm)
    # collapse multi spaces
    text_norm = re.sub(r"\s+", " ", text_norm).strip()

    found = set()

    # 1) add checked items (they are already explicit)
    if checked_list:
        for it in checked_list:
            if it:
                found.add(it.lower().strip())

    # 2) check synonyms: if synonym phrase present in text -> add canonical
    syn_map = kb.get("synonyms", {})
    # sort synonyms by length (longer first) to avoid partial matches
    for s in sorted(syn_map.keys(), key=lambda x: -len(x)):
        patt = r"\b" + re.escape(s.lower()) + r"\b"
        if re.search(patt, text_norm):
            found.add(syn_map[s].lower())

    # 3) check explicit red-flag keywords (exact phrase)
    for rf in kb.get("red_flag_keywords", []):
        patt = r"\b" + re.escape(rf.lower()) + r"\b"
        if re.search(patt, text_norm):
            found.add(rf.lower())

    # 4) check conditions' required/supporting/red words (multi-word phrases)
    # build a set of candidate phrases from KB
    phrases = set()
    for cond in kb.get("conditions", []):
        for field in ("required_symptoms", "supporting_symptoms", "red_flags"):
            for phrase in cond.get(field, []):
                phrases.add(phrase.lower())

    # match longer phrases first
    for phrase in sorted(phrases, key=lambda x: -len(x)):
        patt = r"\b" + re.escape(phrase) + r"\b"
        if re.search(patt, text_norm):
            found.add(phrase)

    # 5) also add single tokens from text if they match any KB token (fallback)
    tokens = text_norm.split()
    kb_tokens = set()
    for p in phrases:
        kb_tokens.update(p.split())
    for t in tokens:
        if t in kb_tokens:
            found.add(t)

    return set(found)

def run_rule_engine(parsed_symptoms, duration_text=None, severity=None, kb=KB):
    """
    Returns:
      - ranked list of conditions with normalized 'score' between 0 and 1
      - final_urgency ('self_care', 'see_gp', 'urgent')
      - explanation details (list of dicts with matched items)
    """
    parsed_set = set([s.lower() for s in parsed_symptoms])
    raw_scores = []
    explanations = []

    # compute raw scores
    for cond in kb["conditions"]:
        cond_name = cond["name"]
        score = WEIGHT_BASE
        matches = {"required": [], "supporting": [], "red_flags": []}

        # required
        for req in cond.get("required_symptoms", []):
            r = req.lower()
            if r in parsed_set:
                score += WEIGHT_REQUIRED
                matches["required"].append(r)

        # supporting
        for sup in cond.get("supporting_symptoms", []):
            s = sup.lower()
            if s in parsed_set:
                score += WEIGHT_SUPPORTING
                matches["supporting"].append(s)

        # red flags
        for rf in cond.get("red_flags", []):
            rfw = rf.lower()
            if rfw in parsed_set:
                score += WEIGHT_RED
                matches["red_flags"].append(rfw)

        raw_scores.append({"condition": cond_name, "raw_score": score,
                           "matches": matches,
                           "recommended_tests": cond.get("recommended_tests", []),
                           "declared_urgency": cond.get("urgency", "see_gp")})

    # normalize raw_score to 0..1
    scores_vals = [r["raw_score"] for r in raw_scores]
    max_raw = max(scores_vals) if scores_vals else 1.0
    min_raw = min(scores_vals) if scores_vals else 0.0

    # avoid divide by zero
    span = max_raw - min_raw if max_raw != min_raw else 1.0
    for r in raw_scores:
        r["score"] = (r["raw_score"] - min_raw) / span
        # round for display
        r["score"] = round(r["score"], 3)

    # sort descending
    ranked = sorted(raw_scores, key=lambda x: x["score"], reverse=True)

    # Decide final urgency:
    # - If any matched red_flag anywhere -> urgent
    # - Else if top declared_urgency is 'urgent' or 'see_gp' and score > threshold -> escalate
    # - Else self-care
    final_urgency = "self_care"
    # check red flags globally
    global_red = False
    for r in ranked:
        if r["matches"]["red_flags"]:
            global_red = True
            break
    if global_red:
        final_urgency = "urgent"
    else:
        top = ranked[0]
        if top["declared_urgency"] == "urgent" and top["score"] >= 0.35:
            final_urgency = "urgent"
        elif top["declared_urgency"] == "see_gp" and top["score"] >= 0.25:
            final_urgency = "see_gp"
        else:
            final_urgency = "self_care"

    # prepare top-3
    top3 = []
    for r in ranked[:3]:
        top3.append({
            "condition": r["condition"],
            "score": r["score"],
            "matches": r["matches"],
            "recommended_tests": r["recommended_tests"],
            "declared_urgency": r["declared_urgency"]
        })

    return top3, final_urgency, {"ranked_all": ranked}

@app.route("/", methods=["GET"])
def index():
    common_symptoms = sorted({
        s for cond in KB["conditions"]
        for s in cond.get("required_symptoms", []) + cond.get("supporting_symptoms", [])
    })
    return render_template("index.html", common_symptoms=common_symptoms)


@app.route("/submit", methods=["POST"])
def submit():
    text = request.form.get("symptoms_text", "")
    duration = request.form.get("duration", "")
    severity = request.form.get("severity", "")
    age = request.form.get("age", "")
    sex = request.form.get("sex", "")
    # collect checkboxes
    checked = request.form.getlist("symptoms_check")

    # handle file
    uploaded_fn = None
    file = request.files.get("image")
    if file and file.filename != "" and allowed_file(file.filename):
        filename = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        uploaded_fn = filename
    elif file and file.filename != "":
        flash("Uploaded file type not allowed. Allowed: png/jpg/jpeg/gif")

    # parse text into keywords
    parsed_from_text = extract_keywords(text, checked_list=checked, kb=KB)
    # combine with checkboxes
    combined = parsed_from_text
    # run rule engine
    top_conditions, urgency, debug = run_rule_engine(list(combined), duration, severity, kb=KB)

    session_id = uuid.uuid4().hex
    result_obj = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "input": {
            "text": text,
            "checked": checked,
            "duration": duration,
            "severity": severity,
            "age": age,
            "sex": sex,
            "image": uploaded_fn
        },
        "parsed_symptoms": list(combined),
        "top_conditions": top_conditions,
        "final_urgency": urgency,
        #"red_flags": red_flags
    }

    # store session (in-memory); for real app persist to DB
    SESSIONS[session_id] = result_obj

    return redirect(url_for("result", session_id=session_id))



@app.route("/result/<session_id>", methods=["GET"])
def result(session_id):
    res = SESSIONS.get(session_id)
    if not res:
        return "Session not found", 404
    return render_template("result.html", r=res)


@app.route("/export/<session_id>", methods=["GET"])
def export(session_id):
    res = SESSIONS.get(session_id)
    if not res:
        return "Session not found", 404
    # Create a small json file to send
    fname = f"report_{session_id}.json"
    path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return send_file(path, as_attachment=True, download_name=fname)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0" , port=8000)
