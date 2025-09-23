import os
import math
from typing import List, Tuple, Dict, Any

from flask import Flask, render_template, jsonify

# Supabase (Python client v2)
from supabase import create_client, Client

# Gemini (Google Generative AI) for embeddings
import google.generativeai as genai
from dotenv import load_dotenv


# Hardcoded Supabase credentials as requested
HARDCODED_SUPABASE_URL = "https://daiuefevbtkthizqykez.supabase.co"
HARDCODED_SUPABASE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRhaXVlZmV2YnRrdGhpenF5a2V6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg1Njg1ODIsImV4cCI6MjA3NDE0NDU4Mn0.S8K7oRe2aTAa9ci0BlBZ26GOAn0O7WCFFwdl5o8gi-Y"
)


def create_app() -> Flask:
    app = Flask(__name__)

    # Load environment variables from .env if present
    load_dotenv()

    # Initialize Supabase client
    app.config["SUPABASE_URL"] = HARDCODED_SUPABASE_URL
    app.config["SUPABASE_KEY"] = HARDCODED_SUPABASE_KEY
    app.config["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY", "AIzaSyAsutlZP3xvpyGEMiRmWa-8GKhLaKjxb34")

    app.supabase: Client = create_client(
        app.config["SUPABASE_URL"], app.config["SUPABASE_KEY"]
    )

    # Configure Gemini
    if app.config["GEMINI_API_KEY"]:
        genai.configure(api_key=app.config["GEMINI_API_KEY"])

    @app.route("/")
    def index():
        return jsonify({
            "status": "ok",
            "routes": ["/student_matching"],
        })

    @app.route("/student_matching")
    def student_matching():
        try:
            students = fetch_students(app.supabase)
        except Exception as exc:  # pragma: no cover - surface error to UI
            return render_template(
                "student_matching.html",
                error=f"Failed to fetch students: {exc}",
                students=[],
                match_pair=None,
                similarity=None,
                match_reason=None,
            )

        if not students:
            return render_template(
                "student_matching.html",
                error="No students found.",
                students=[],
                match_pair=None,
                similarity=None,
                match_reason=None,
            )

        # If no Gemini key present, show message
        if not app.config["GEMINI_API_KEY"]:
            return render_template(
                "student_matching.html",
                error=(
                    "GEMINI_API_KEY not set. Create a .env with GEMINI_API_KEY and restart."
                ),
                students=students,
                match_pair=None,
                similarity=None,
                match_reason=None,
            )

        try:
            match_pair, similarity = match_best_pair_with_gemini(students)
            match_reason = generate_match_reason(match_pair[0], match_pair[1])
        except Exception as exc:  # pragma: no cover
            return render_template(
                "student_matching.html",
                error=f"Failed to compute matches: {exc}",
                students=students,
                match_pair=None,
                similarity=None,
                match_reason=None,
            )

        return render_template(
            "student_matching.html",
            error=None,
            students=students,
            match_pair=match_pair,
            similarity=similarity,
            match_reason=match_reason,
        )

    @app.route("/dashboard")
    def dashboard():
        try:
            students = fetch_students(app.supabase)
        except Exception as exc:  # pragma: no cover - surface error to UI
            return render_template(
                "dashboard.html",
                error=f"Failed to fetch students: {exc}",
                overall_summary=None,
                categories=[],
                suggestions=[],
            )

        if not students:
            return render_template(
                "dashboard.html",
                error="No students found.",
                overall_summary=None,
                categories=[],
                suggestions=[],
            )

        texts = [s["summary"] for s in students]

        if not app.config["GEMINI_API_KEY"]:
            # Fallback: simple keyword categories without model
            insights = fallback_insights(texts)
        else:
            try:
                insights = generate_institute_insights(texts)
            except Exception:
                insights = fallback_insights(texts)

        # Normalize categories into list of {label, count, percent}
        raw_categories = ensure_multi_categories(insights.get("categories", []) or [], texts)
        total_counts = sum(int(cat.get("count", 0)) for cat in raw_categories)
        categories = []
        if total_counts > 0:
            for cat in raw_categories:
                count = int(cat.get("count", 0))
                percent = (count / total_counts) * 100.0 if total_counts else 0.0
                categories.append({
                    "label": cat.get("label", "Other"),
                    "count": count,
                    "percent": percent,
                })
        else:
            # Ensure we always have something to show
            categories = [{"label": "General", "count": len(texts), "percent": 100.0 if texts else 0.0}]

        # Suggestions fallback
        suggestions = insights.get("suggestions", []) or []
        if not suggestions:
            suggestions = fallback_insights(texts).get("suggestions", [])

        # Points formatting for overall summary
        overall_text = insights.get("overall_summary") or ""
        overall_points = format_overall_summary_to_points(overall_text)

        # Derived KPIs
        categories_identified = len(categories)
        top_concern = insights.get("main_issue") or (max(categories, key=lambda x: x.get("count", 0)).get("label") if categories else extract_main_issue_from_summary(" ".join(overall_points)))

        return render_template(
            "dashboard.html",
            error=None,
            overall_points=overall_points or insights.get("insight_points", []),
            categories=categories,
            suggestions=suggestions,
            categories_identified=categories_identified,
            top_concern=top_concern,
        )

    return app


def fetch_students(supabase: Client) -> List[Dict[str, Any]]:
    """Fetch students with summaries from Supabase.

    Expected table: chat_summaries(id, student_name, summary, diagnoses, created_at)
    """
    # Adjust table/columns here if your schema differs
    response = supabase.table("chat_summaries").select(
        "student_name, summary"
    ).execute()
    data = response.data or []

    # Normalize to required fields
    normalized: List[Dict[str, Any]] = []
    for row in data:
        if row.get("summary"):
            normalized.append(
                {
                    "name": row.get("student_name") or "Student",
                    "summary": row.get("summary"),
                }
            )
    return normalized


def embed_texts_with_gemini(texts: List[str]) -> List[List[float]]:
    """Return embeddings for list of texts using Gemini embedding model."""
    # Use the recommended text embedding model
    embeddings: List[List[float]] = []
    for text in texts:
        # Use top-level embed_content helper as recommended
        result = genai.embed_content(model="models/text-embedding-004", content=text)
        # Normalize various possible response shapes
        values: List[float] = []
        if isinstance(result, dict):
            if "embedding" in result:
                emb = result["embedding"]
                if isinstance(emb, dict):
                    values = emb.get("values") or emb.get("value") or []
                elif isinstance(emb, list):
                    values = emb
            elif "data" in result:
                # Some SDKs wrap embeddings in result['data'][0]['embedding']
                data = result.get("data") or []
                if isinstance(data, list) and data:
                    first = data[0] or {}
                    emb = first.get("embedding")
                    if isinstance(emb, dict):
                        values = emb.get("values") or emb.get("value") or []
                    elif isinstance(emb, list):
                        values = emb
        else:
            # Fallback for object-style response
            emb_obj = getattr(result, "embedding", None)
            if isinstance(emb_obj, list):
                values = emb_obj
            else:
                values = getattr(emb_obj, "values", None) or getattr(emb_obj, "value", None) or []
        embeddings.append(list(values))
    return embeddings


def summarize_texts_with_gemini(texts: List[str]) -> List[str]:
    """Summarize each input text using Gemini 2.5 Flash to standardize content before embedding."""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    summaries: List[str] = []
    for text in texts:
        prompt = (
            "Summarize the following student's description in 2-3 concise sentences, "
            "focusing on key traits, interests, and support needs relevant for pairing.\n\n"
            f"Text:\n{text}"
        )
        try:
            resp = model.generate_content(prompt)
            summarized = getattr(resp, "text", None) or ""
        except Exception:
            summarized = text  # Fallback to original if generation fails
        summaries.append(summarized.strip() or text)
    return summaries


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def match_best_pair_with_gemini(
    students: List[Dict[str, Any]]
) -> Tuple[Tuple[Dict[str, Any], Dict[str, Any]], float]:
    """Find the most similar pair of students based on summary embeddings."""
    if len(students) < 2:
        raise ValueError("Need at least two students to match")

    texts = [s["summary"] for s in students]
    # First, normalize/summarize using Gemini 2.5 Flash, then embed the summaries
    summarized_texts = summarize_texts_with_gemini(texts)
    vectors = embed_texts_with_gemini(summarized_texts)

    best_i, best_j, best_score = 0, 1, -1.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            score = cosine_similarity(vectors[i], vectors[j])
            if score > best_score:
                best_score = score
                best_i, best_j = i, j

    return (students[best_i], students[best_j]), best_score


def generate_match_reason(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    """Generate a concise, friendly reason why two students match well."""
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    prompt = (
        "Given two student summaries, explain in 1-2 short sentences why they make a good match. "
        "Be specific but concise, referencing complementary interests, goals, styles, or support needs.\n\n"
        f"Student A (name: {a.get('name','Student A')}):\n{a.get('summary','')}\n\n"
        f"Student B (name: {b.get('name','Student B')}):\n{b.get('summary','')}\n\n"
        "Response:"
    )
    try:
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "")
        return (text or "They share compatible interests and goals, making them a strong match.").strip()
    except Exception:
        return "They share compatible interests and goals, making them a strong match."


def generate_institute_insights(texts: List[str]) -> Dict[str, Any]:
    """Use Gemini to create an overall summary, category distribution, and suggestions.

    Returns shape: {
      overall_summary: str,
      insight_points: [str],
      categories: [{label: str, count: int}],
      suggestions: [str],
      main_issue: str
    }
    """
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    joined = "\n\n".join(texts)
    prompt = (
        "You will analyze many anonymous student mental health summaries from one institute.\n"
        "1) Produce a concise overall summary (3-5 sentences) of common themes, stressors, strengths.\n"
        "1b) Provide 4-6 punchy bullet insights (7-12 words each) as 'insight_points'.\n"
        "2) Create 4-7 category buckets with counts (estimate), focusing on mental health topics like anxiety, depression, stress, academic pressure, social challenges, burnout, wellbeing. Return as JSON field 'categories'.\n"
        "3) Return the single most prominent issue as 'main_issue' (short phrase, e.g., 'Academic Pressure').\n"
        "4) Provide 4-6 actionable suggestions for institute-wide support (e.g., workshops, group therapy, peer circles, mindfulness, accommodations).\n"
        "Return strictly JSON with keys: overall_summary, main_issue, insight_points (list), categories (list of {label, count}), suggestions (list).\n\n"
        f"Summaries (anonymous):\n{joined}\n\n"
        "JSON:"
    )
    resp = model.generate_content(prompt)
    text = getattr(resp, "text", "{}")
    # Defensive parse
    import json as _json  # local import to avoid polluting global namespace
    try:
        data = _json.loads(text)
    except Exception:
        data = {"overall_summary": text, "main_issue": "", "insight_points": [], "categories": [], "suggestions": []}
    # Ensure shapes
    if not isinstance(data.get("categories"), list):
        data["categories"] = []
    if not isinstance(data.get("suggestions"), list):
        data["suggestions"] = []
    if not isinstance(data.get("insight_points"), list):
        data["insight_points"] = []
    if not isinstance(data.get("main_issue"), str):
        data["main_issue"] = ""
    return data


def fallback_insights(texts: List[str]) -> Dict[str, Any]:
    """Simple heuristic insights if Gemini is unavailable."""
    lowered = "\n".join(texts).lower()
    keywords = {
        "Anxiety": ["anxiety", "anxious", "nervous"],
        "Depression": ["depression", "depressed", "sad"],
        "Stress": ["stress", "stressed", "overwhelmed"],
        "Academic Pressure": ["exam", "grades", "assignments", "academic"],
        "Burnout": ["burnout", "exhausted", "fatigue"],
        "Social": ["social", "friends", "lonely", "isolation"],
        "Wellbeing": ["wellbeing", "well-being", "mindfulness", "sleep"],
    }
    counts: Dict[str, int] = {k: 0 for k in keywords}
    for label, terms in keywords.items():
        counts[label] = sum(lowered.count(t) for t in terms)
    categories = [{"label": k, "count": v} for k, v in counts.items() if v > 0]
    if not categories:
        categories = [{"label": "General", "count": len(texts)}]
    suggestions = [
        "Offer weekly group therapy focused on stress and coping skills",
        "Host peer support circles for social connection",
        "Run mindfulness and sleep hygiene workshops",
        "Provide academic time-management and study strategy sessions",
        "Create a confidential check-in channel with counselors",
    ]
    overall_summary = (
        "Students report varied mental health concerns with recurring themes of stress, academic "
        "pressure, and social challenges. A mix of preventative workshops and peer spaces, "
        "paired with accessible counseling, may improve overall wellbeing."
    )
    insight_points = [
        "Stress and academic pressure appear frequently across cohorts",
        "Social connection and belonging are recurring support needs",
        "Sleep and energy challenges hint at emerging burnout risks",
        "Skill-building workshops can improve coping and study habits",
    ]
    return {"overall_summary": overall_summary, "main_issue": "Stress", "insight_points": insight_points, "categories": categories, "suggestions": suggestions}


def format_overall_summary_to_points(summary_text: str) -> List[str]:
    """Turn a freeform summary or JSON-like text into neat bullet points."""
    if not summary_text:
        return []
    # If summary accidentally contains JSON, strip braces/quotes crudely and split sentences
    cleaned = summary_text.replace("{", " ").replace("}", " ").replace("\n", " ")
    # Split by periods and semicolons
    parts = [p.strip().strip('- ').strip('"\'') for p in cleaned.replace(';', '.').split('.')]
    points = [p for p in parts if len(p) > 0]
    # Keep 3-6 punchy points
    return points[:6]


def derive_keyword_categories(texts: List[str]) -> List[Dict[str, Any]]:
    lowered = "\n".join(texts).lower()
    keywords = {
        "Stress": ["stress", "stressed", "overwhelmed"],
        "Academic Pressure": ["exam", "grades", "assignments", "academic"],
        "Anxiety": ["anxiety", "anxious", "nervous"],
        "Depression": ["depression", "depressed", "sad"],
        "Social": ["social", "friends", "lonely", "isolation"],
        "Burnout": ["burnout", "exhausted", "fatigue"],
        "Wellbeing": ["wellbeing", "well-being", "mindfulness", "sleep"],
    }
    counts: Dict[str, int] = {}
    for label, terms in keywords.items():
        counts[label] = sum(lowered.count(t) for t in terms)
    cats = [{"label": k, "count": v} for k, v in counts.items() if v > 0]
    if not cats:
        cats = [{"label": "Stress", "count": max(1, len(texts)//3)}, {"label": "Academic Pressure", "count": max(1, len(texts)//3)}, {"label": "Social", "count": max(1, len(texts) - 2*max(1, len(texts)//3))}]
    cats.sort(key=lambda c: c["count"], reverse=True)
    return cats[:7]


def ensure_multi_categories(categories: List[Dict[str, Any]], texts: List[str]) -> List[Dict[str, Any]]:
    # If we only have one category or empty, derive more from keywords
    nonzero = [c for c in categories if c.get("count", 0) > 0]
    if len(nonzero) >= 2:
        return categories
    derived = derive_keyword_categories(texts)
    return derived


def extract_main_issue_from_summary(summary_text: str) -> str:
    if not summary_text:
        return "Stress"
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = (
            "From the following institute-wide mental health summary, extract the SINGLE most prominent issue as a 1-3 word label (e.g., 'Academic Pressure').\n"
            f"Summary:\n{summary_text}\n\nLabel only:"
        )
        resp = model.generate_content(prompt)
        label = (getattr(resp, "text", "") or "").strip()
        return label.split("\n")[0][:60] or "Stress"
    except Exception:
        return "Stress"


if __name__ == "__main__":
    # For local development
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
