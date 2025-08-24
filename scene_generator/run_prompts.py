import json
import sys
from pathlib import Path

# Import your existing agent from main.py
# Make sure main.py defines a function `make_app()` that returns a compiled LangGraph app
from generator_agent import make_app

# -------------------------------
# Config
# -------------------------------
PROMPTS_FILE = "scene_prompts/prompts.json"  # your JSON with list of prompts
MODEL = "gpt-4o-mini"  # optional, override default model in main.py

# -------------------------------
# Load prompts
# -------------------------------
with open(PROMPTS_FILE, "r") as f:
    data = json.load(f)

prompts = data.get("prompts", [])
if not prompts:
    print("No prompts found in JSON.")
    sys.exit(1)

# -------------------------------
# Initialize the agent
# -------------------------------
app = make_app(model=MODEL)

# -------------------------------
# Loop through prompts and invoke agent
# -------------------------------
for i, prompt in enumerate(prompts, start=1):
    print(f"⚡ Generating floor plan for prompt {i}: {prompt}")
    try:
        app.invoke({"prompt": prompt})
    except Exception as e:
        print(f"❌ Error generating prompt {i}: {e}")
