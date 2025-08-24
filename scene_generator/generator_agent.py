import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Optional: load .env if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# -----------------------------
# Configuration
# -----------------------------
ROOT_DIR = Path("generated_floor_plans")
ROOT_DIR.mkdir(exist_ok=True)

ALLOWED_ROOM_TYPES = {
    "kitchen",
    "bedroom",
    "living-room",
    "closet",
    "hallway",
    "bathroom",
    "garage",
    "balcony",
    "dining-room",
    "utility",
    "staircase-room",
    "warehouse",
    "office",
    "meeting-room",
    "open-office",
    "break-room",
    "restroom",
    "factory-office",
}

SYSTEM_INSTRUCTION = """
You are an AI architect assistant.

GOAL:
Produce a JSON floorplan for EXACTLY ONE room (not an entire apartment/floor). Output MUST be valid JSON only (no commentary, no markdown, no trailing text).

REQUIRED JSON SHAPE:
{
  "rooms": {
    "<roomtype>_0/0": {
      "shape": "shapely.Polygon([...])" | "shapely.box(xmin,ymin,xmax,ymax)"
    }
  },
  "doors": {
    "door": {"shape": "shapely.LineString([...])"},
    "door.001": {"shape": "shapely.LineString([...])"}
  },
  "interiors": {
    "interior": {"shape": "shapely.LineString([...])"},
    "interior.001": {"shape": "shapely.LineString([...])"}
  },
  "windows": {
    "window": {"shape": "shapely.LineString([...])"},
    "window.001": {"shape": "shapely.LineString([...])", "is_panoramic": 1}
  }
  // NOTE: include "opens" ONLY if you actually add openings:
  // "opens": {
  //   "open": {"shape": "shapely.LineString([...])"},
  //   "open.001": {"shape": "shapely.LineString([...])"}
  // }
}

NAMING & MAPPINGS:
- Room type key MUST be one of:
  kitchen, bedroom, living-room, closet, hallway, bathroom, garage, balcony,
  dining-room, utility, staircase-room, warehouse, office, meeting-room,
  open-office, break-room, restroom, factory-office
- The single room key MUST follow numbering format "<roomtype>_0/0" (e.g., "kitchen_0/0").
- Element keys should follow: "door" then "door.001", "door.002", ...
  Same pattern for windows ("window", "window.001", ...),
  interiors ("interior", "interior.001", ...),
  and opens ("open", "open.001", ...).
- Include empty categories ONLY if needed; "opens" is OPTIONAL.

GEOMETRY RULES:
- Use Shapely-like strings exactly as shown (do not return objects).
- Rooms can be rectangles (shapely.box) OR non-rectangular polygons (shapely.Polygon).
- Vary sizes realistically; don’t always produce rectangles.
- Doors/windows should lie on room boundary segments.
- Interiors are line segments inside the room (e.g., counters/partitions).
- Coordinates can be any consistent unit; keep them numerically reasonable.

STRICTNESS:
- Output MUST contain EXACTLY ONE entry in "rooms".
- Output MUST be valid JSON. No extra keys beyond rooms/doors/interiors/windows and optional opens.
- If a section has no items, you MAY omit it.

REFERENCE EXAMPLE (FORMAT ONLY):
{
  "rooms": {
    "kitchen_0/0": {
      "shape": "shapely.Polygon([(0,0),(6,0),(6,4),(0,4)])"
    }
  },
  "doors": {
    "door": {
      "shape": "shapely.LineString([(3,0),(4,0)])"
    }
  },
  "interiors": {
    "interior": {
      "shape": "shapely.LineString([(2,4),(4,4)])"
    }
  },
  "windows": {
    "window": {
      "shape": "shapely.LineString([(6,1),(6,3)])"
    },
    "window.001": {
      "shape": "shapely.LineString([(1,4),(2,4)])",
      "is_panoramic": 1
    }
  }
}
"""


# -----------------------------
# Helpers
# -----------------------------
def next_indexed_filename(room_dir: Path, room_type: str) -> Path:
    room_dir.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in room_dir.glob(f"{room_type}_*.json"):
        m = re.search(rf"{re.escape(room_type)}_(\d+)\.json$", p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return room_dir / f"{room_type}_{max_n + 1:03d}.json"


def validate_single_room_plan(plan: Dict[str, Any]) -> None:
    if not isinstance(plan, dict):
        raise ValueError("Plan is not a JSON object.")
    if "rooms" not in plan or not isinstance(plan["rooms"], dict):
        raise ValueError('Missing "rooms" object.')
    if len(plan["rooms"]) != 1:
        raise ValueError('Plan must contain EXACTLY one room in "rooms".')

    room_key = next(iter(plan["rooms"].keys()))
    if "_0/0" not in room_key:
        raise ValueError('Room key must follow "<roomtype>_0/0" format.')

    room_type = room_key.split("_")[0]
    if room_type not in ALLOWED_ROOM_TYPES:
        raise ValueError(f'Room type "{room_type}" not allowed.')

    # Minimal checks for geometry strings
    room_obj = plan["rooms"][room_key]
    if not isinstance(room_obj, dict) or "shape" not in room_obj:
        raise ValueError('Room must include a "shape" string.')
    shape = room_obj["shape"]
    if not isinstance(shape, str) or not shape.startswith("shapely."):
        raise ValueError('Room "shape" must be a shapely.* string.')

    # Optional sections: doors, windows, interiors, opens
    for sec, required_prefix in [
        ("doors", "shapely.LineString"),
        ("windows", "shapely.LineString"),
        ("interiors", "shapely.LineString"),
        ("opens", "shapely.LineString"),
    ]:
        if sec in plan:
            if not isinstance(plan[sec], dict):
                raise ValueError(f'"{sec}" must be an object.')
            for k, v in plan[sec].items():
                if (
                    not isinstance(v, dict)
                    or "shape" not in v
                    or not isinstance(v["shape"], str)
                ):
                    raise ValueError(f'Every {sec} entry must have a string "shape".')
                if not v["shape"].startswith(required_prefix):
                    raise ValueError(
                        f'{sec} "{k}" shape must start with {required_prefix}(...).'
                    )


def save_plan(plan: Dict[str, Any]) -> Path:
    # Get the only room key
    room_key = next(iter(plan["rooms"].keys()))
    room_type = room_key.split("_")[0]
    room_dir = ROOT_DIR / room_type
    out_path = next_indexed_filename(room_dir, room_type)
    with open(out_path, "w") as f:
        json.dump(plan, f, indent=2)
    return out_path


# -----------------------------
# Build the graph
# -----------------------------
def make_app(model: str = "gpt-4o-mini"):

    llm = ChatOpenAI(model=model, temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

    def generate(state: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = state["prompt"]
        resp = llm.invoke(
            [
                SystemMessage(content=SYSTEM_INSTRUCTION),
                HumanMessage(content=user_prompt),
            ]
        )
        raw = resp.content.strip()
        try:
            plan = json.loads(raw)
        except Exception as e:
            raise ValueError(
                f"LLM did not return valid JSON.\nError: {e}\nOutput:\n{raw}"
            )
        return {"plan": plan}

    def validate_and_save(state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state["plan"]
        validate_single_room_plan(plan)
        # Normalize: allow missing optional sections by leaving them out.
        out_path = save_plan(plan)
        print(f"✅ Saved: {out_path}")
        return {"path": str(out_path)}

    graph = StateGraph(dict)
    graph.add_node("generate", generate)
    graph.add_node("validate_and_save", validate_and_save)
    graph.set_entry_point("generate")
    graph.add_edge("generate", "validate_and_save")
    graph.add_edge("validate_and_save", END)
    return graph.compile()


# -----------------------------
# CLI
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python main.py "Generate a kitchen with an L-shaped polygon, one door, two windows" [--model gpt-4o-mini]'
        )
        sys.exit(1)

    # simple arg parsing
    model = "gpt-4o-mini"
    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        if i + 1 < len(sys.argv):
            model = sys.argv[i + 1]
            prompt = " ".join(sys.argv[1:i])
        else:
            print("Error: --model requires a value")
            sys.exit(1)
    else:
        prompt = " ".join(sys.argv[1:])

    app = make_app(model=model)
    app.invoke({"prompt": prompt})


if __name__ == "__main__":
    main()
