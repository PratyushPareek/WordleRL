"""CPU-only check of the training setup: config builds, tool schema, and the exact prompt text
Qwen3 sees with thinking on and off. Downloads only the tokenizer.

  python scripts/check_setup.py [--model Qwen/Qwen3-4B]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer
from transformers.utils import get_json_schema

from wordle_rl.env import WordleToolEnv
from wordle_rl.prompt import build_prompt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    args = p.parse_args()

    env = WordleToolEnv()
    schema = get_json_schema(env.guess)
    print("Tool schema shown to the model:\n" + json.dumps(schema, indent=2))

    tok = AutoTokenizer.from_pretrained(args.model)
    env.reset(secret="slate")
    reply = env.guess("crane")
    messages = build_prompt() + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "guess", "arguments": {"word": "crane"}}}]},
        {"role": "tool", "name": "guess", "content": reply},
    ]
    for thinking in (False, True):
        text = tok.apply_chat_template(
            messages, tools=[env.guess], tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
        print(f"\n===== enable_thinking={thinking}: prompt after one guess ({len(tok(text).input_ids)} tokens) =====")
        print(text)


if __name__ == "__main__":
    main()
