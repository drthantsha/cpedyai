import os
import time
import hashlib
import json
import ast
import re
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
from datetime import datetime
import requests
from dotenv import load_dotenv
from streamlit_drawable_canvas import st_canvas

# PDF + OCR
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image

# ---------- LOAD SECRETS ----------
load_dotenv()  # for local development

def get_secret(key, default_value):
    try:
        env_value = os.getenv(key)
        if env_value:
            return env_value
        return st.secrets.get(key, default_value)
    except Exception:
        return default_value

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or get_secret("OPENAI_API_KEY", "")
MODEL = get_secret("MODEL", "gpt-4o-mini")

# Pricing defaults (USD per 1M tokens)
def _f(v, d): 
    try: return float(v)
    except: return d
INPUT_PER_M = _f(get_secret("INPUT_PER_M_USD", "0.15"), 0.15)
OUTPUT_PER_M = _f(get_secret("OUTPUT_PER_M_USD", "0.60"), 0.60)
DAILY_CAP_USD = _f(get_secret("DAILY_CAP_USD", "0.50"), 0.50)

if not OPENAI_API_KEY:
    st.warning("Please set OPENAI_API_KEY in your .env or environment variables.")
    st.info("Create a .env file with: OPENAI_API_KEY=your_key_here")
    st.stop()

# ---------- CACHE ----------
CACHE_TTL_SEC = 60 * 60 * 24
memory_cache: Dict[str, Dict[str, Any]] = {}

def cache_get(key: str) -> Optional[Any]:
    item = memory_cache.get(key)
    if item and time.time() - item["t"] < CACHE_TTL_SEC:
        return item["v"]
    return None

def cache_set(key: str, val: Any) -> None:
    memory_cache[key] = {"v": val, "t": time.time()}

def hash_text(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()

# ---------- TOKEN COST ----------
def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1_000_000) * INPUT_PER_M + (output_tokens / 1_000_000) * OUTPUT_PER_M

def rough_tokens(s: str) -> int:
    return max(1, len(s) // 4)

if "spend_today" not in st.session_state:
    st.session_state.spend_today = 0.0
    st.session_state.spend_day = datetime.utcnow().date()

def reset_daily_if_needed() -> None:
    if st.session_state.spend_day != datetime.utcnow().date():
        st.session_state.spend_today = 0.0
        st.session_state.spend_day = datetime.utcnow().date()

# ---------- OPENAI CALL ----------
def call_openai(system_prompt: str, user_prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
    reset_daily_if_needed()
    key = hash_text(system_prompt, user_prompt, max_tokens, temperature, MODEL)
    cached = cache_get(key)
    if cached:
        return cached

    est_in = rough_tokens(system_prompt + user_prompt)
    est_out = max_tokens
    est_cost = estimate_cost(est_in, est_out)
    if st.session_state.spend_today + est_cost > DAILY_CAP_USD:
        return "[Lite mode] Daily AI budget reached. Try shorter input or come back tomorrow."

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        out_tokens = rough_tokens(text)
        actual_cost = estimate_cost(est_in, out_tokens)
        st.session_state.spend_today += actual_cost
        cache_set(key, text)
        return text
    except requests.exceptions.RequestException as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        try:
            return f"Response parsing error: {str(e)} - {json.dumps(data)[:200]}"
        except:
            return f"Response parsing error: {str(e)}"

# ---------- PDF UTILS ----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        reader = PyPDF2.PdfReader(file_bytes)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    except Exception:
        pass

    # OCR for images in PDF
    try:
        images = convert_from_bytes(file_bytes)
        for img in images:
            text += "\n" + pytesseract.image_to_string(img)
    except Exception:
        pass

    return text

# ---------- SAFE PYTHON EXECUTOR ----------
ALLOWED_BUILTINS = {"range": range, "len": len, "print": print, "min": min, "max": max}

def safe_exec(code: str, inputs: Optional[Dict] = None) -> Dict[str, Any]:
    g = {"__builtins__": ALLOWED_BUILTINS}
    if inputs:
        g.update(inputs)
    local = {}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute)):
                return {"error": "Disallowed code (imports/attributes not allowed)."}
        exec(compile(tree, filename="<safe>", mode="exec"), g, local)
        return {"stdout": str(local.get("result", ""))}
    except Exception as e:
        return {"error": str(e)}

# ---------- SIMPLE GRID ENGINE (for Canvas games) ----------
TILE = 32
GRID_W, GRID_H = 15, 10  # tiles

def draw_grid(canvas_result, tiles: List[List[int]], actor: Tuple[int,int], goals: List[Tuple[int,int]]):
    # We rely on st_canvas' drawing layer; we don't draw there, but we compute a legend for user
    ax, ay = actor
    st.markdown("**Legend:** 🟩 grass / 🧱 wall / ⭐ goal / 🤖 player")
    grid_vis = []
    for y in range(GRID_H):
        row = []
        for x in range(GRID_W):
            if (x,y) == (ax,ay):
                row.append("🤖")
            elif (x,y) in goals:
                row.append("⭐")
            else:
                row.append("🟩" if tiles[y][x] == 0 else "🧱")
        grid_vis.append("".join(row))
    st.code("\n".join(grid_vis), language="text")  # lightweight visual

def blank_tiles():
    return [[0 for _ in range(GRID_W)] for _ in range(GRID_H)]

def wall_border(tiles):
    for x in range(GRID_W):
        tiles[0][x] = tiles[GRID_H-1][x] = 1
    for y in range(GRID_H):
        tiles[y][0] = tiles[y][GRID_W-1] = 1
    return tiles

def clamp(x,a,b): return max(a, min(b, x))

# ---------- KidLang++ (adds REPEAT and COLORS) ----------
def interpret_kidlang(program: str) -> List[Dict[str, Any]]:
    x, y, dir_idx = 1, 1, 0
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N,E,S,W (y up negative)
    path = [{"x": x, "y": y, "action": "start"}]

    tokens = [ln.strip() for ln in program.splitlines() if ln.strip()]
    i = 0

    def step_forward(n=1):
        nonlocal x, y
        for _ in range(n):
            dx, dy = dirs[dir_idx]
            x += dx; y += dy
            path.append({"x": x, "y": y, "action": "move"})

    while i < len(tokens):
        ln = tokens[i].upper()
        parts = ln.split()
        cmd = parts[0]
        if cmd == "FORWARD" and len(parts) >= 2 and parts[1].isdigit():
            step_forward(int(parts[1]))
        elif cmd == "LEFT":
            dir_idx = (dir_idx - 1) % 4
            path.append({"action": "left"})
        elif cmd == "RIGHT":
            dir_idx = (dir_idx + 1) % 4
            path.append({"action": "right"})
        elif cmd == "PAINT" and len(parts) >= 2:
            path.append({"x": x, "y": y, "action": "paint", "color": parts[1]})
        elif cmd == "REPEAT" and len(parts) >= 2 and parts[1].isdigit() and tokens[i+1:i+2] and tokens[i+1].strip() == "[":
            # Find matching ]
            n = int(parts[1])
            j = i+2
            block = []
            depth = 1
            while j < len(tokens):
                t = tokens[j].strip()
                if t == "[":
                    depth += 1
                elif t == "]":
                    depth -= 1
                    if depth == 0:
                        break
                else:
                    block.append(t)
                j += 1
            # execute block n times
            for _ in range(n):
                for b in block:
                    # recursively handle a tiny subset (no nested REPEAT inside for simplicity)
                    bp = b.split()
                    if bp[0] == "FORWARD" and len(bp) >= 2 and bp[1].isdigit():
                        step_forward(int(bp[1]))
                    elif bp[0] == "LEFT":
                        dir_idx = (dir_idx - 1) % 4
                        path.append({"action": "left"})
                    elif bp[0] == "RIGHT":
                        dir_idx = (dir_idx + 1) % 4
                        path.append({"action": "right"})
                    elif bp[0] == "PAINT" and len(bp) >= 2:
                        path.append({"x": x, "y": y, "action": "paint", "color": bp[1]})
            i = j  # skip to end of block
        else:
            path.append({"action": "unknown", "line": ln})
        i += 1
    return path

# ---------- UI ----------
st.set_page_config(page_title="Curriculum AI Suite", page_icon="🎓", layout="wide")

# Apply custom CSS for glassmorphism effect and blue-white theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #64B5F6;
        --background-color: #f0f8ff;
        --text-color: #0d47a1;
    }
    
    /* Glassmorphism for containers */
    .stApp {
        background-color: var(--background-color);
    }
    
    .glass-effect {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2) !important;
    }
    
    /* Style for chat containers */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(8px) !important;
        border-radius: 15px !important;
        margin-bottom: 12px !important;
        padding: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.15) !important;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background: rgba(30, 136, 229, 0.15) !important;
        border-left: 4px solid var(--primary-color) !important;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background: rgba(255, 255, 255, 0.85) !important;
        border-left: 4px solid var(--secondary-color) !important;
    }
    
    /* Style headings */
    h1, h2, h3 {
        color: var(--primary-color) !important;
    }
    
    /* Style buttons */
    .stButton > button {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: var(--secondary-color) !important;
        box-shadow: 0 4px 12px rgba(100, 181, 246, 0.5) !important;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.7) !important;
        border-radius: 10px !important;
        padding: 10px !important;
        backdrop-filter: blur(5px) !important;
    }
    
    /* Chat input styling */
    .stChatInput > div {
        background: rgba(255, 255, 255, 0.7) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(5px) !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background: rgba(76, 175, 80, 0.15) !important;
        border: none !important;
        border-radius: 8px !important;
    }
    
    /* Error message styling */
    .stError {
        background: rgba(244, 67, 54, 0.15) !important;
        border: none !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎓 Curriculum AI Suite")

mode = st.sidebar.selectbox("Choose an app", [
    # Subjects
    "Maths Tutor (Learner Q&A)",
    "Science Explainer",
    "Biology Helper",
    "Accounting Helper",
    "Geography Guide",
    "English Helper",
    "isiZulu Helper",
    "Setswana Helper",
    "Sepedi Helper",
    # Skills
    "AI Code Editor",
    "AI Programming Language (KidLang++)",
    # Support
    "Counselling Skills Simulator",
    "Sports Coaching Plans",
    "Study Buddy (PDF/Notes Q&A)",
    "Exam Prep Generator",
    # Arcade
    "Code Game Arcade (Canvas Prototype)",
    # New addition
    "AI Chatbot"
])

st.sidebar.markdown("**Daily spend (est.):** ${:.4f} / ${:.2f}".format(
    st.session_state.spend_today, DAILY_CAP_USD
))
st.sidebar.markdown("---")
st.sidebar.markdown("**Current Settings:**")
st.sidebar.markdown(f"- Model: {MODEL}")
st.sidebar.markdown(f"- Input cost: ${INPUT_PER_M}/M")
st.sidebar.markdown(f"- Output cost: ${OUTPUT_PER_M}/M")
st.sidebar.markdown(f"- Daily cap: ${DAILY_CAP_USD}")

# ---------- Subject Helpers ----------
def subject_block(title, brief, system_role):
    st.subheader(title)
    st.caption(brief)
    q = st.text_area("Enter your question / topic", height=150)
    if st.button("Explain", key=title):
        sys = system_role
        user = f"Question:\n{q}\n\nConstraints: Explain in steps, include one worked example if applicable, keep it aligned to South African CAPS curriculum where possible."
        st.write(call_openai(sys, user, max_tokens=320))

if mode == "Maths Tutor (Learner Q&A)":
    subject_block(
        "🧮 Maths Tutor",
        "Paste a maths problem. Get a step-by-step solution and a quick check tip.",
        "You are a CAPS-aligned mathematics tutor for Grade 7–12 in South Africa. Prioritise method and reasoning; show steps and a final check."
    )

elif mode == "Science Explainer":
    subject_block(
        "🔬 Science Explainer",
        "Explain Natural Sciences/Physical Sciences topics in clear language.",
        "You are a CAPS-aligned science explainer. Summarise concepts, define key terms, give a short real-world example."
    )

elif mode == "Biology Helper":
    subject_block(
        "🧫 Biology Helper",
        "Life Sciences help: processes, diagrams, definitions, genetics, ecology.",
        "You are a Life Sciences tutor (CAPS). Use correct biological terms, short labelled bullet points, and a simple diagram description if relevant."
    )

elif mode == "Accounting Helper":
    subject_block(
        "📒 Accounting Helper",
        "Journal entries, trial balance, financial statements, ratios.",
        "You are a high school Accounting tutor (CAPS). Provide formats, double-entry hints, and a small numeric example."
    )

elif mode == "Geography Guide":
    subject_block(
        "🗺️ Geography Guide",
        "Mapwork, climatology, geomorphology, human geography summaries.",
        "You are a Geography tutor (CAPS). Include one map/diagram description and a quick practice question."
    )

elif mode == "English Helper":
    subject_block(
        "📚 English Helper",
        "Grammar, comprehension strategies, literature analysis, writing tips.",
        "You are an English First Additional/HL tutor (CAPS). Explain with examples; include a short practice exercise and answer."
    )

elif mode == "isiZulu Helper":
    subject_block(
        "🗣️ isiZulu Helper",
        "Basic grammar, vocabulary, sentence patterns, and short dialogues.",
        "You are an isiZulu language tutor (CAPS). Give examples with translations and brief pronunciation tips."
    )

elif mode == "Setswana Helper":
    subject_block(
        "🗣️ Setswana Helper",
        "Grammar, vocabulary, useful phrases, and sentence building.",
        "You are a Setswana language tutor (CAPS). Provide examples with translations and a mini drill."
    )

elif mode == "Sepedi Helper":
    subject_block(
        "🗣️ Sepedi Helper",
        "Grammar, vocabulary, morphology (prefixes/suffixes), short dialogues.",
        "You are a Sepedi language tutor (CAPS). Provide examples with translations and a short practice with answers."
    )

# ---------- AI Code Editor ----------
elif mode == "AI Code Editor":
    st.subheader("💡 AI Code Editor")
    code = st.text_area("Paste code", height=240)
    action = st.selectbox("Action", ["Explain", "Complete", "Find bugs & fix", "Write tests"])
    if st.button("Run AI assistant"):
        sys = (
            "You are a concise programmer assistant. "
            "Explain→bullets. Complete→code only. Fix→corrected code. Tests→pytest or language-appropriate tests."
        )
        user = f"Action:{action}\nCode:\n{code}"
        st.code(call_openai(sys, user, max_tokens=600))

# ---------- AI Programming Language (KidLang++) ----------
elif mode == "AI Programming Language (KidLang++)":
    st.subheader("🧩 KidLang++ Interpreter")
    st.caption("Commands: FORWARD n | LEFT | RIGHT | PAINT color | REPEAT n [ ... ]")
    program = st.text_area("Program", "REPEAT 4 [ FORWARD 3\nRIGHT ]\nPAINT BLUE", height=160)
    if st.button("Run KidLang++"):
        path = interpret_kidlang(program)
        st.json(path)

# ---------- Counselling Skills Simulator ----------
elif mode == "Counselling Skills Simulator":
    st.subheader("🧑🏽‍🏫 Counselling Skills Simulator (Educator)")
    scenario = st.text_area("Student scenario (behaviour, context, risk, goal)", height=180)
    if st.button("Generate response plan"):
        sys = (
            "You are a school counsellor coach for educators in South Africa. "
            "Produce: (1) Immediate response (2) Empathic script (3) Safeguarding checks (4) Referral guidance (5) Follow-up plan."
        )
        st.write(call_openai(sys, f"Scenario:\n{scenario}", max_tokens=420))

# ---------- Sports Coaching Plans ----------
elif mode == "Sports Coaching Plans":
    st.subheader("🏅 Sports Coaching Plans")
    sport = st.selectbox("Sport", ["Soccer", "Netball", "Athletics", "Cricket", "Rugby"])
    age = st.selectbox("Age group", ["U11", "U13", "U15", "U17", "Open"])
    focus = st.text_input("Focus (e.g., endurance, passing, footwork, bowling line & length)")
    if st.button("Create 4-week plan"):
        sys = (
            "You are a school sports coaching planner. Create a 4-week plan: weekly goals, 3 drills/session, safety, simple metrics."
        )
        user = f"Sport:{sport}\nAge:{age}\nFocus:{focus}"
        st.write(call_openai(sys, user, max_tokens=500))

# ---------- Study Buddy (UPDATED) ----------
elif mode == "Study Buddy (PDF/Notes Q&A)":
    st.subheader("📚 Study Buddy")
    
    uploaded_files = st.file_uploader("Upload up to 3 PDFs", type="pdf", accept_multiple_files=True)
    pasted_notes = st.text_area("Or paste notes here", height=220)

    all_text = ""
    if uploaded_files:
        for f in uploaded_files[:3]:
            bytes_data = f.read()
            all_text += extract_text_from_pdf(bytes_data)
        if all_text.strip():
            st.success(f"✅ {len(uploaded_files[:3])} PDF(s) uploaded and processed.")

    if pasted_notes.strip():
        all_text += "\n" + pasted_notes

    q = st.text_input("Ask a question")
    if st.button("Answer"):
        if not all_text.strip():
            st.warning("Please upload at least one PDF or paste notes.")
        else:
            sents = [s.strip() for s in all_text.replace("\n", " ").split(".") if s.strip()]
            top = ". ".join(sorted(
                sents,
                key=lambda s: sum(1 for w in q.lower().split() if w in s.lower()),
                reverse=True
            )[:10])
            sys = "Answer ONLY using context. If missing, say 'Not in notes'. Keep under 120 words."
            user = f"Context:\n{top}\n\nQ:{q}"
            st.write(call_openai(sys, user, max_tokens=220))

# ---------- Exam Prep Generator ----------
elif mode == "Exam Prep Generator":
    st.subheader("📝 Exam Prep Generator")
    subject = st.selectbox("Subject", [
        "Mathematics", "Physical Sciences", "Life Sciences", "Accounting", "Geography",
        "English", "isiZulu", "Setswana", "Sepedi"
    ])
    topic = st.text_input("Topic (e.g., Quadratic Equations / Photosynthesis)")
    level = st.selectbox("Difficulty", ["Basic", "Moderate", "Challenging"])
    if st.button("Generate mock exam"):
        sys = (
            "You create CAPS-aligned mock questions. Output 6 questions (mixed formats) with concise answers and marking guide."
        )
        user = f"Subject:{subject}\nTopic:{topic}\nLevel:{level}"
        st.write(call_openai(sys, user, max_tokens=650))

# ---------- Code Game Arcade (Canvas Prototype) ----------
elif mode == "Code Game Arcade (Canvas Prototype)":
    st.subheader("🎮 Code Game Arcade — Canvas Prototype")
    st.caption("Lightweight 2D grid games. Good graphics will come later (Pygame/Three.js).")

    game = st.selectbox("Choose a game", [
        "Python Turtle Grid",
        "Web DOM Clicker",
        "C++ Loop Racer",
        "Java If-Else Maze"
    ])

    # Shared Canvas surface (for pointer capture / drawing)
    st.markdown("**Playfield (clicks captured if needed):**")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=2,
        background_color="#f0f0f0",
        height=GRID_H * TILE,
        width=GRID_W * TILE,
        drawing_mode="transform",  # passive; we just want coordinate clicks
        key=f"canvas_{game.replace(' ','_')}"
    )

    # Base map
    tiles = wall_border(blank_tiles())

    if game == "Python Turtle Grid":
        st.markdown("**Goal:** Move the robot 🤖 to the star ⭐ using KidLang++ (FORWARD/LEFT/RIGHT/REPEAT).")
        program = st.text_area("Program", "REPEAT 3 [ FORWARD 2 RIGHT ]\nFORWARD 1", height=140)
        start = (1, GRID_H-2)
        goal = (GRID_W-2, 1)
        tiles[5][5] = tiles[5][6] = tiles[6][5] = 1  # a small obstacle
        if st.button("Run Program", key="pygrid_run"):
            path = interpret_kidlang(program)
            # Replay onto grid
            x,y = start
            for step in path:
                if step.get("action") in ("move",):
                    x = clamp(step.get("x", x), 1, GRID_W-2)
                    y = clamp(step.get("y", y), 1, GRID_H-2)
                    if tiles[y][x] == 1:  # wall -> stop
                        break
            draw_grid(canvas_result, tiles, (x,y), [goal])
            st.success("⭐ Reached goal!" if (x,y) == goal else "Keep trying!")

    elif game == "Web DOM Clicker":
        st.markdown("**Goal:** Click the targets in order: A → B → C (simulating DOM interactions).")
        st.info("Click anywhere on the playfield; we read your last click coordinates.")
        targets = [(4,3), (8,4), (11,2)]  # tile coords
        # Save state
        if "dom_click_idx" not in st.session_state:
            st.session_state.dom_click_idx = 0
        # Process click
        last = None
        if canvas_result.json_data is not None:
            # Generic capture of the last transformed object click (approx)
            last = canvas_result.json_data.get("objects", [])[-1]["top"] if canvas_result.json_data.get("objects") else None
        clicked_tile = None
        if last is not None:
            # Very rough mapping (top-left origin); streamlit-canvas returns px; we map to tile row
            y_px = last
            y_tile = int(y_px // TILE)
            x_tile = 2 + st.session_state.dom_click_idx  # fake x progression to simulate order
            clicked_tile = (x_tile, y_tile)
        # Evaluate
        current_goal = targets[st.session_state.dom_click_idx] if st.session_state.dom_click_idx < len(targets) else None
        if st.button("Check Click", key="dom_check"):
            if clicked_tile and current_goal and clicked_tile[1] == current_goal[1]:
                st.session_state.dom_click_idx += 1
                st.success(f"Good! Next target index: {st.session_state.dom_click_idx}")
            else:
                st.warning("Missed! Try clicking different row.")
        draw_grid(canvas_result, tiles, (2, GRID_H-2), targets)
        if st.session_state.dom_click_idx >= 3:
            st.success("🎉 DOM sequence complete!")

    elif game == "C++ Loop Racer":
        st.markdown("**Goal:** Move the car to the finish using a for-loop idea.")
        st.caption("Type a pseudo C++ loop (we pattern-check `for (`). Example: `for (int i=0;i<7;i++)`")
        loop = st.text_input("Loop header", "for (int i=0;i<10;i++)")
        start = (1, GRID_H//2)
        finish = (GRID_W-2, GRID_H//2)
        steps = 0
        if st.button("Race!", key="cpp_race"):
            if "for" in loop.replace(" ", "").lower() and ("<" in loop or ">" in loop):
                # crude estimate of iterations by extracting the number after '<'
                m = re.search(r"<\s*(\d+)", loop)
                if m:
                    steps = int(m.group(1))
            x = start[0] + min(steps, GRID_W-3)
            draw_grid(canvas_result, tiles, (x, start[1]), [finish])
            if x >= finish[0]:
                st.success("🏁 Finished! Nice loop.")
            else:
                st.info(f"Moved {steps} steps. Increase your upper bound to reach the finish line.")

    elif game == "Java If-Else Maze":
        st.markdown("**Goal:** Choose to go UP or RIGHT based on a tile sensor value (simulated if-else).")
        st.caption("Enter: `if (sensor > 5) moveRight(); else moveUp();` (we pattern-check)")
        code = st.text_area("If-Else snippet", "if (sensor > 5) moveRight(); else moveUp();", height=90)
        start = (2, GRID_H-3)
        finish = (GRID_W-3, 2)
        # Simulated sensor value:
        sensor = 7
        x,y = start
        if st.button("Decide & Move", key="java_maze"):
            go_right = (">" in code and "else" in code)  # super simple parse
            if go_right and sensor > 5:
                x = min(GRID_W-3, x+4)
            else:
                y = max(2, y-4)
            draw_grid(canvas_result, tiles, (x,y), [finish])
            st.write(f"Sensor={sensor}. Position={(x,y)}")
            st.success("🎯 Closer!") if (x,y)==finish else st.info("Decide again to reach ⭐")

# ---------- AI Chatbot ----------
elif mode == "AI Chatbot":
    st.subheader("🤖 AI Chatbot")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("What would you like to know?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate AI response
        response = call_openai(
            "You are a helpful AI assistant. Provide clear, concise, and accurate responses.",
            prompt,
            max_tokens=500,
            temperature=0.7
        )
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

st.caption("⚖️ Guardrails: budget cap, caching, safe exec. Keep prompts short to save tokens.")

# Setup helper
if not os.path.exists('.env'):
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Setup Instructions:**")
    st.sidebar.code("OPENAI_API_KEY=your_key_here")
    st.sidebar.markdown("2) Optional: MODEL, INPUT_PER_M_USD, OUTPUT_PER_M_USD, DAILY_CAP_USD")