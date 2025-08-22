#!/usr/bin/env python3
"""
oai_reasoning.py — REPL for GPT-5 reasoning models (Responses API).
Server-side conversation tracked via previous_response_id.

Behavior:
- REPL until /quit or Ctrl-C (one-shot if piped or --one-shot).
- /reset clears server-side conversation.
- No "Assistant:" prefix; just raw outputs.
"""

import os, sys, argparse
from typing import Optional
from openai import OpenAI

DEFAULT_MODEL = "gpt-5-mini"

def call_reasoning(
    client: OpenAI,
    model: str,
    user_input: str,
    system_prompt: str,
    budgets: list[int],
    effort: str,
    prev_id: Optional[str],
    stream: bool,
) -> tuple[str, Optional[str]]:
    """
    Make a reasoning-model request (GPT-5 family) with token budget escalation.
    Tries each budget in order until a complete response is returned.
    Returns (assistant_text, new_response_id).
    """
    last_error = None

    for max_out in budgets:
        kwargs = dict(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            max_output_tokens=max_out,
            reasoning={"effort": effort},
        )
        if prev_id:
            kwargs["previous_response_id"] = prev_id

        try:
            if stream:
                chunks: list[str] = []
                new_id = None
                with client.responses.stream(**kwargs) as stream_obj:
                    for event in stream_obj:
                        if event.type == "response.output_text.delta":
                            print(event.delta, end="", flush=True)
                            chunks.append(event.delta)
                        elif event.type == "response.completed":
                            new_id = event.response.id
                    print("", flush=True)
                return "".join(chunks).strip(), new_id

            else:
                resp = client.responses.create(**kwargs)
                text = resp.output_text.strip()
                new_id = resp.id

                # Check if it stopped because of max tokens
                data = resp.to_dict() if hasattr(resp, "to_dict") else {}
                status = data.get("status")
                reason = (data.get("incomplete_details") or {}).get("reason")
                if status == "incomplete" and reason == "max_output_tokens":
                    last_error = "max_output_tokens"
                    continue  # try next budget

                return text, new_id

        except Exception as e:
            last_error = str(e)
            continue

    # If no attempt succeeded
    return (f"[Error: {last_error or 'no response'}]", None)


def run_repl(args):
    client = OpenAI()
    prev_id = None

    print(f"oai reasoning REPL ({args.model}) started.")
    print("Commands: /quit, /reset\n")

    try:
        while True:
            user = input("\nYou: ").strip()
            if not user:
                continue
            if user.lower() in ("/quit", "/exit"):
                break
            if user.lower() == "/reset":
                prev_id = None
                print("(conversation reset)")
                continue

            text, new_id = call_reasoning(
                client=client,
                model=args.model,
                user_input=user,
                system_prompt=args.system,
                budgets=args.budgets,     # ✅ fixed
                effort=args.effort,
                prev_id=prev_id,
                stream=args.stream,
            )
            if not args.stream:
                print(text)

            prev_id = new_id or prev_id

    except KeyboardInterrupt:
        print("\n(^C) exiting session")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL)
    ap.add_argument("--system", default="Answer directly and concisely.")
    ap.add_argument("--budgets", default="800,1600",
                    help="Comma-separated list of max_output_tokens attempts")
    ap.add_argument("--effort", choices=["low","medium","high"], default="medium")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--one-shot", action="store_true")
    args = ap.parse_args()

    # Parse budgets into a list of ints
    args.budgets = [int(x) for x in args.budgets.split(",") if x.strip().isdigit()] or [800, 1600]

    client = OpenAI()
    if args.one_shot or not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("No prompt provided.")
            sys.exit(1)
        text, _ = call_reasoning(
            client=client,
            model=args.model,
            user_input=prompt,
            system_prompt=args.system,
            budgets=args.budgets,   # ✅ fixed
            effort=args.effort,
            prev_id=None,
            stream=args.stream,
        )
        if not args.stream:
            print(text)
    else:
        run_repl(args)


if __name__ == "__main__":
    main()
