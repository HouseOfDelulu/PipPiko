# app.py
#
# One program that does everything: loads your Qwen model, serves the chat
# UI, and answers its API calls — all from the same process.
#
# Install:
#   pip install -r requirements.txt
#
# Run:
#   python app.py
#
# Then just wait — it opens http://localhost:5000 in your browser
# automatically once the server is up (the page itself waits for the model
# to finish loading before unlocking the chat).

import ast
import difflib
import json
import re
import threading
import webbrowser

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

PORT = 5000

# ---------------------------------------------------------------------------
# Model loading — same model/setup as your original script. Loaded once, in
# a background thread, so the web server can come up immediately and the
# page can show a loading screen while this finishes.
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
_model = None
_tokenizer = None
_model_ready = False
_model_error = None


def load_model():
    global _model, _tokenizer, _model_ready, _model_error
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading {MODEL_NAME} ...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype="auto", device_map="auto"
        )
        print("Model loaded. GPU available:", torch.cuda.is_available())
        _model_ready = True
    except Exception as exc:  # noqa: BLE001
        _model_error = str(exc)
        print("Failed to load model:", _model_error)


def qwen(
    prompt: str,
    system: str = "You are a friendly shopping assistant.",
    do_sample: bool = False,
) -> str:
    """Same generation call as the original script.

    `do_sample` exists because greedy decoding is deterministic: the same
    context produces the exact same sentence every time, which is how the bot
    ended up asking one question over and over. Retries sample instead.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    sampling = {"do_sample": True, "temperature": 0.8, "top_p": 0.9} if do_sample else {"do_sample": False}
    outputs = _model.generate(**inputs, max_new_tokens=200, **sampling)
    response = _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response.strip()


# ---------------------------------------------------------------------------
# Prompt templates — same rules as your original script, parameterized so
# the conversation comes from the frontend on each request instead of being
# kept in a global Python list.
# ---------------------------------------------------------------------------
DIALOG_SYSTEM_PROMPT = "You are Pip, a friendly shopping assistant."

DIALOG_PROMPT_TEMPLATE = """You are a friendly shopping assistant.

Your goal is to understand what product the user wants.

Have a natural conversation with the user.

Provide recommendations when appropriate.

If the user gives unrealistic answers, correct them.

BUDGET RULE:
- You MUST ask the user for their actual budget amount.
- Ask for a numerical amount, e.g. "What's the maximum amount you'd like to spend?"
- Do NOT simply ask whether they have a budget.
- If the user says they don't know, ask them once for an approximate amount.
- If they still cannot or will not give a number, accept that and move on to
  a different question. Never ask about the budget more than twice.
- Ask for the budget as the second question.

AFTER THE BUDGET, COVER THESE TWO:
- Brand: ask whether they have a brand in mind, and make clear that "any" is a
  fine answer. If they have no preference, accept it and do not ask again.
- Usage: ask what they need it for, phrased for the product in question. For
  phones, laptops, monitors and other electronics, ask whether it is for
  personal use or for work and office use. For clothing, shoes and accessories,
  ask whether they will be wearing it indoors or outdoors. Never ask about
  terrain or surfaces for electronics.

NEVER REPEAT YOURSELF:
- Do not ask a question you have already asked, even reworded.
- If the user has already answered or declined something, move to a new topic.

IMPORTANT RULES:
1. Ask only ONE question at a time.
2. Never ask multiple questions in one message.
3. Do not list all the information you need.
4. Do not repeat questions that have already been answered.
5. Use the previous conversation to decide what to ask next.
6. Ask useful questions that help narrow down the product.
7. You can acknowledge the user's answer before asking the next question.
8. Always remember to ask the user for their budget amount.
8. When you have enough information to understand what the user wants,
   respond with exactly:
SEARCH_READY

Conversation so far:

{convo}

What should you say next?"""

EXTRACT_PROMPT_TEMPLATE = """Convert this shopping conversation into a Python dictionary.

Conversation:

{convo}

Use these fields:

category
brand
usage
terrain
budget

Rules:

1. Only use information explicitly stated by the user.
2. If a field is unknown, use None.
3. budget should be a number if available.
4. Return ONLY the Python dictionary literal. No explanations, no markdown.

Example:

{{"category": "running shoes", "brand": "Nike", "usage": "running", "terrain": "road", "budget": 150}}"""


def conversation_to_text(conversation):
    lines = []
    for msg in conversation:
        speaker = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"{speaker}: {msg.get('text', '')}")
    return "\n".join(lines)


def parse_requirements(raw_text: str) -> dict:
    """Robustly parse the model's dict-ish output, same spirit as the
    original script's ast.literal_eval, with a JSON fallback."""
    match = re.search(r"\{.*\}", raw_text, re.S)
    candidate = match.group(0) if match else raw_text

    for parser in (json.loads, ast.literal_eval):
        try:
            result = parser(candidate)
            if isinstance(result, dict):
                return {
                    "category": result.get("category"),
                    "brand": result.get("brand"),
                    "usage": result.get("usage"),
                    "terrain": result.get("terrain"),
                    "budget": result.get("budget"),
                }
        except Exception:  # noqa: BLE001
            continue

    return {"category": None, "brand": None, "usage": None, "terrain": None, "budget": None}


# ---------------------------------------------------------------------------
# Budget enforcement
#
# A 1.5B model reliably ignores "ask for the budget as the second question",
# and it will happily emit SEARCH_READY before a budget ever comes up. So the
# budget rule is enforced here in Python instead of being left to the prompt:
# we detect whether the user has actually given an amount, and if not we
# override the model's reply with the budget question.
# ---------------------------------------------------------------------------
BUDGET_QUESTION = "Happy to help with that! What's the maximum amount you'd like to spend?"
BUDGET_REASK = (
    "No worries if you're not sure — roughly what number should I stay under? "
    "A ballpark amount is fine."
)

# "$300", "300 dollars", "1.5k", "around 250 bucks" — an amount is only
# accepted when it carries a currency marker, so counts like "2 people" or
# "size 10" are not mistaken for a budget.
_MARKED_AMOUNT_RE = re.compile(
    r"""(?:
            [$€£₹]\s*(?P<sym>\d[\d,]*(?:\.\d+)?)\s*(?P<symk>k\b)?
          | (?P<word>\d[\d,]*(?:\.\d+)?)\s*(?P<wordk>k\b)?\s*
            (?:dollars?|bucks?|usd|eur|gbp|inr|rupees?|quid)
          | (?P<bare>\d[\d,]*(?:\.\d+)?)\s*k\b
        )""",
    re.I | re.X,
)

# A bare number ("300", "around 1200") only counts as a budget when it is the
# answer to a budget question.
_BARE_AMOUNT_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(k\b)?", re.I)

_BUDGET_ASK_RE = re.compile(r"budget|spend|afford|price range|how much", re.I)


def _coerce_number(value):
    """Turn the model's budget field into a number, or None."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        match = re.search(r"\d[\d,]*(?:\.\d+)?", value)
        if match:
            return float(match.group(0).replace(",", ""))
    return None


def _to_amount(digits: str, thousands: bool) -> float:
    value = float(digits.replace(",", ""))
    return value * 1000 if thousands else value


def _amount_in(text: str, allow_bare: bool):
    """Pull a budget amount out of one user message, or None."""
    match = _MARKED_AMOUNT_RE.search(text)
    if match:
        for digits_key, k_key in (("sym", "symk"), ("word", "wordk"), ("bare", None)):
            digits = match.group(digits_key)
            if digits:
                return _to_amount(digits, bool(k_key and match.group(k_key)) or digits_key == "bare")

    if allow_bare:
        match = _BARE_AMOUNT_RE.search(text)
        if match:
            return _to_amount(match.group(1), bool(match.group(2)))

    return None


def find_budget(conversation):
    """Latest budget amount the user has given, or None.

    Currency-marked amounts are accepted anywhere in the conversation; a bare
    number is only accepted when the assistant's previous message was asking
    about budget.
    """
    found = None
    previous_bot_text = ""

    for msg in conversation:
        text = str(msg.get("text", ""))
        if msg.get("role") == "user":
            amount = _amount_in(text, allow_bare=bool(_BUDGET_ASK_RE.search(previous_bot_text)))
            if amount is not None:
                found = amount
        else:
            previous_bot_text = text

    return found


def budget_prompt_for(conversation) -> str:
    """Ask for the budget, without repeating the exact same sentence twice."""
    last_bot = next(
        (m.get("text", "") for m in reversed(conversation) if m.get("role") != "user"), ""
    )
    return BUDGET_REASK if _BUDGET_ASK_RE.search(str(last_bot)) else BUDGET_QUESTION


# ---------------------------------------------------------------------------
# Loop breaking
#
# Two separate causes of the bot repeating itself:
#
#   1. Greedy decoding (do_sample=False) is deterministic, so when the user's
#      new message barely changes the context, the model emits a byte-for-byte
#      identical question.
#   2. The prompt says to keep asking for a budget until an amount arrives, so
#      "I don't spend on electronics" produced an infinite budget loop.
#
# So we detect repeats, retry with sampling, and fall back to a fixed question
# bank. And a refusal counts as a settled answer instead of being ignored.
# ---------------------------------------------------------------------------
MAX_BUDGET_ASKS = 2
REPEAT_SIMILARITY = 0.82

_REFUSAL_RE = re.compile(
    r"\b(?:"
    r"i\s+do\s?n[o']?t\s+(?:know|spend|care|mind|have)"
    r"|do\s?n[o']?t\s+know"
    r"|no\s+idea|not\s+sure|unsure|no\s+budget|no\s+preference|not\s+really"
    r"|does\s?n[o']?t\s+matter|no\s+limit|any(?:thing)?\s+is\s+fine"
    r"|whatever|you\s+decide|you\s+choose|up\s+to\s+you|skip|i\s+told\s+you"
    r"|first\s+time|never\s+bought"
    r")\b",
    re.I,
)

# Which slot a question is about, so we can tell "asked about budget again"
# from "asked something new".
_TOPIC_PATTERNS = [
    ("budget", _BUDGET_ASK_RE),
    ("brand", re.compile(r"\bbrand|manufacturer|make\b", re.I)),
    ("usage", re.compile(
        r"use it for|using it for|\busage|purpose|mainly use|what will you"
        r"|personal use|office use|work or office|work and office"
        r"|wear(?:ing)?\s+(?:it|this|them)|indoors or outdoors", re.I)),
    ("terrain", re.compile(r"terrain|surface|indoor|outdoor|\broad\b|\btrail", re.I)),
]

MOVING_ON = "No problem — I'll work without that. "

# ---------------------------------------------------------------------------
# Required slots
#
# Same problem as the budget, same solution: the model will emit SEARCH_READY
# after two answers and never get round to brand or usage, so we ask for them
# here instead of trusting the prompt. Budget stays separate above because it
# has its own re-ask limit and refusal handling.
# ---------------------------------------------------------------------------
REQUIRED_SLOTS = ("brand", "usage")

BRAND_QUESTION = (
    "Do you have a particular brand in mind, or shall I keep it open to any?"
)

# A usage question only earns its turn if it's phrased for the thing being
# bought. "Indoors or outdoors" is meaningless for a monitor, and "personal or
# office" is meaningless for a pair of jeans.
USAGE_QUESTIONS = {
    "tech": "Will this be mainly for personal use, or for work and office use?",
    "apparel": "Will you mostly be wearing this indoors or outdoors?",
    None: "What will you mainly use it for?",
}

# Whole words that place what the user asked for into one of those groups.
# Checked in order, so "smartwatch" lands in tech before "watch" can claim it
# for apparel.
_CATEGORY_GROUPS = [
    ("tech", [
        "phone", "phones", "smartphone", "smartphones", "mobile", "iphone",
        "android", "laptop", "laptops", "notebook", "macbook", "computer",
        "pc", "desktop", "monitor", "monitors", "display", "screen", "tablet",
        "ipad", "headphones", "headset", "earbuds", "earphones", "airpods",
        "smartwatch", "console", "camera", "keyboard", "mouse", "tv",
    ]),
    ("apparel", [
        "shoe", "shoes", "sneaker", "sneakers", "trainers", "boot", "boots",
        "sandals", "heels", "jacket", "coat", "jeans", "pants", "trousers",
        "chinos", "shorts", "leggings", "sweater", "hoodie", "shirt", "dress",
        "skirt", "clothes", "clothing", "outfit", "jewelry", "necklace",
        "earrings", "bag", "backpack", "watch",
    ]),
]


def _normalize_reply(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).strip()


def bot_messages(conversation):
    return [str(m.get("text", "")) for m in conversation if m.get("role") != "user"]


def count_budget_asks(conversation) -> int:
    return sum(1 for text in bot_messages(conversation) if _BUDGET_ASK_RE.search(text))


def budget_refused(conversation) -> bool:
    """Has the user ever declined to name a budget?

    Sticky on purpose. Checking only the most recent message meant that one
    vaguely-worded turn ("still not telling you") re-opened a question the
    user had already refused twice.
    """
    asked_budget = False

    for msg in conversation:
        text = str(msg.get("text", ""))
        if msg.get("role") == "user":
            if asked_budget and _REFUSAL_RE.search(text):
                return True
        else:
            asked_budget = bool(_BUDGET_ASK_RE.search(text))

    return False


def is_repeat(reply: str, conversation) -> bool:
    """True if this reply is one the assistant has effectively already sent."""
    candidate = _normalize_reply(reply)
    if not candidate:
        return False

    for previous in bot_messages(conversation):
        earlier = _normalize_reply(previous)
        if not earlier:
            continue
        if earlier == candidate:
            return True
        if difflib.SequenceMatcher(None, earlier, candidate).ratio() >= REPEAT_SIMILARITY:
            return True

    return False


def generate_reply(conversation, avoid_repeats: bool = False, do_sample: bool = False) -> str:
    prompt = DIALOG_PROMPT_TEMPLATE.format(convo=conversation_to_text(conversation))
    if avoid_repeats:
        prompt += (
            "\n\nIMPORTANT: You have already asked the questions above. Do NOT "
            "repeat or rephrase any question you have already asked. Ask about "
            "something different, or reply SEARCH_READY if you have enough."
        )
    return qwen(prompt, system=DIALOG_SYSTEM_PROMPT, do_sample=do_sample)


def category_group(conversation):
    """Which family of product the user is shopping for, or None if unclear."""
    said = " ".join(
        _normalize_reply(m.get("text", ""))
        for m in conversation
        if m.get("role") == "user"
    )
    padded = f" {said} "
    for group, keywords in _CATEGORY_GROUPS:
        if any(f" {word} " in padded for word in keywords):
            return group
    return None


def question_for(topic, conversation) -> str:
    if topic == "brand":
        return BRAND_QUESTION
    return USAGE_QUESTIONS[category_group(conversation)]


def slot_answered(topic, conversation) -> bool:
    """True once we've asked about `topic` and the user has said something back.

    Any reply counts, "no preference" included — the test is whether the user
    had their chance to answer, not whether the slot ended up filled. Without
    that, declining a brand would put us straight back to asking about brands.
    """
    pattern = dict(_TOPIC_PATTERNS)[topic]
    asked_at = None

    for i, msg in enumerate(conversation):
        if msg.get("role") == "user":
            continue
        if pattern.search(str(msg.get("text", ""))):
            asked_at = i

    if asked_at is None:
        return False
    return any(m.get("role") == "user" for m in conversation[asked_at + 1:])


# What the model hands back when the user said they don't mind. All of these
# mean the same thing to the ranking — no brand filter — so they get collapsed
# into one label the panel can show.
_OPEN_BRAND_VALUES = {
    "any", "any brand", "anything", "none", "no brand", "no preference",
    "open", "unknown", "not specified", "n/a", "na", "whatever", "no idea",
    "doesn't matter", "does not matter", "no", "flexible",
}

ANY_BRAND = "Any"


def resolve_brand(extracted, conversation):
    """The brand to show and rank on: a real name, "Any", or nothing yet.

    A blank brand has two very different meanings — "we haven't asked yet" and
    "the user doesn't mind" — and the panel showed a dash for both. Once the
    question has been asked and answered, an empty extraction means the search
    is genuinely brand-agnostic, so say so.
    """
    name = str(extracted or "").strip()

    if name and name.lower() not in _OPEN_BRAND_VALUES:
        return name
    return ANY_BRAND if slot_answered("brand", conversation) else None


def next_required_question(conversation, prefix: str = ""):
    """The next slot we insist on covering, or None once they're all covered."""
    for topic in REQUIRED_SLOTS:
        if not slot_answered(topic, conversation):
            return prefix + question_for(topic, conversation)
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    if _model_ready:
        return jsonify({"status": "ready"})
    if _model_error:
        return jsonify({"status": "error", "detail": _model_error}), 500
    return jsonify({"status": "loading"}), 503


@app.route("/api/chat", methods=["POST"])
def chat():
    if not _model_ready:
        return jsonify({"error": "Model is still loading"}), 503

    data = request.get_json(force=True) or {}
    conversation = data.get("conversation", [])
    force_finish = bool(data.get("force_finish", False))

    if force_finish:
        return jsonify({"reply": "SEARCH_READY"})

    budget = find_budget(conversation)
    user_turns = sum(1 for m in conversation if m.get("role") == "user")
    budget_asks = count_budget_asks(conversation)

    # A budget is "settled" once we have an amount, or once the user has told
    # us they can't or won't give one. Asking a third time is what made the
    # bot feel broken, so we take no for an answer.
    budget_settled = (
        budget is not None
        or budget_asks >= MAX_BUDGET_ASKS
        or budget_refused(conversation)
    )

    # The budget is question #2, always. The greeting already asked what
    # they're shopping for, so once they've answered that we ask this
    # ourselves rather than hoping the model remembers the rule.
    if not budget_settled and user_turns == 1:
        return jsonify({"reply": BUDGET_QUESTION})

    reply = generate_reply(conversation)

    # Loop breaker 1: the model repeated a question. Retry with sampling to
    # escape greedy determinism, then give up on the model and ask something
    # from the fallback bank.
    if is_repeat(reply, conversation):
        reply = generate_reply(conversation, avoid_repeats=True, do_sample=True)
        if is_repeat(reply, conversation):
            reply = next_required_question(conversation) or "SEARCH_READY"

    finished = bool(re.search(r"SEARCH_READY", reply, re.I))

    # Loop breaker 2: budget is settled but the model is still asking about it.
    if budget_settled and not finished and _BUDGET_ASK_RE.search(reply):
        # Say "moving on" once, not in front of every follow-up question.
        already_said = any(MOVING_ON.strip() in text for text in bot_messages(conversation))
        prefix = MOVING_ON if budget is None and not already_said else ""
        reply = next_required_question(conversation, prefix) or "SEARCH_READY"
        finished = reply == "SEARCH_READY"

    # Don't let the model declare itself done before the budget is settled.
    elif not budget_settled and finished:
        reply = budget_prompt_for(conversation)
        finished = False

    # Brand and usage get the same enforcement as the budget. Left to itself the
    # model answers two questions and calls it a day, so "a phone, around 1.5k"
    # went straight to results with brand and usage never raised.
    if finished:
        reply = next_required_question(conversation) or reply

    return jsonify({"reply": reply})


@app.route("/api/extract", methods=["POST"])
def extract():
    if not _model_ready:
        return jsonify({"error": "Model is still loading"}), 503

    data = request.get_json(force=True) or {}
    conversation = data.get("conversation", [])

    prompt = EXTRACT_PROMPT_TEMPLATE.format(convo=conversation_to_text(conversation))
    raw = qwen(prompt, system=DIALOG_SYSTEM_PROMPT)
    requirements = parse_requirements(raw)

    # The model often drops the budget or returns it as text ("$150", "150").
    # Trust the regex over the model, and always hand the frontend a number
    # so the panel and the price ranking both work.
    requirements["budget"] = find_budget(conversation) or _coerce_number(requirements["budget"])
    requirements["brand"] = resolve_brand(requirements["brand"], conversation)

    return jsonify(requirements)


def open_browser():
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    # Load the model in the background so the server (and the page's
    # loading screen) can come up immediately instead of blocking.
    threading.Thread(target=load_model, daemon=True).start()

    # Open the browser shortly after the server starts.
    threading.Timer(1.0, open_browser).start()

    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
