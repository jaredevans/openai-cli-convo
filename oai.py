#!/usr/bin/env python3
"""
oai.py — Prompt → OpenAI Responses API → print text (GPT-5 Mini default)
- REPL until /quit or Ctrl-C (one-shot if stdin is piped or --one-shot).
- Keeps ONLY user turns in history; no prior assistant content is sent.
- Maintains a running compressed summary injected as a system message.
- Optional streaming; escalates max_output_tokens on "max_output_tokens".
"""

import os, sys, json, time, argparse
from typing import Optional, List, Tuple, Union
from openai import OpenAI

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_HISTORY_TURNS = 10
DEFAULT_SUMMARY_MAX_CHARS = 1200
DEFAULT_SUMMARY_EVERY = 1  # summarize after every exchange

# ----------------- Utilities -----------------

def _resp_to_dict(resp):
    for attr in ("to_dict", "model_dump"):
        if hasattr(resp, attr):
            try:
                return getattr(resp, attr)()
            except Exception:
                pass
    return {}

def _extract_text(resp) -> Optional[str]:
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    data = _resp_to_dict(resp) or {}
    outputs = data.get("output") or data.get("outputs") or []
    for item in outputs:
        for part in item.get("content", []) or []:
            t = part.get("text")
            if isinstance(t, str) and t.strip():
                return t.strip()
            if part.get("type") == "message":
                for mc in part.get("content", []) or []:
                    if mc.get("type") == "text" and isinstance(mc.get("text"), str):
                        t2 = mc["text"].strip()
                        if t2:
                            return t2
    tnode = data.get("text")
    if isinstance(tnode, dict):
        tval = tnode.get("value")
        if isinstance(tval, str) and tval.strip():
            return tval.strip()
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = (((choices[0] or {}).get("message") or {}).get("content") or "")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    return None

def _responses_create(client: OpenAI, **kwargs):
    try:
        return client.responses.create(**kwargs)
    except TypeError:
        if "text" in kwargs:
            kwargs2 = dict(kwargs)
            kwargs2.pop("text", None)
            return client.responses.create(**kwargs2)
        raise

def read_prompt() -> str:
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text
    try:
        return input("Enter your prompt: ").strip()
    except EOFError:
        return ""

# ----------------- Prompt Builders -----------------

def build_messages_only_users(
    user_history: List[str],
    current_user: str,
    instructions: Optional[str],
    running_summary: Optional[str],
    max_pairs: int,
):
    msgs: List[dict] = []
    sys_lines = []
    if instructions:
        sys_lines.append(instructions)
    if running_summary:
        sys_lines.append(f"\nConversation summary so far (for context only, do not repeat verbatim):\n{running_summary}")
    if sys_lines:
        msgs.append({"role": "system", "content": "\n".join(sys_lines)})

    for u in user_history[-max_pairs:]:
        msgs.append({"role": "user", "content": u})
    msgs.append({"role": "user", "content": current_user})
    return msgs

# ----------------- Core Call -----------------

def call_openai_with_escalation(
    input_payload: Union[str, List[dict]],
    *,
    model: str,
    budgets: List[int],
    debug: bool,
    pass_instructions: Optional[str] = None,
) -> Optional[str]:
    """
    input_payload: either a plain string or a messages[] list.
    If you already embedded instructions in messages, leave pass_instructions=None.
    """
    if not os.getenv("OPENAI_API_KEY"):
        if debug:
            sys.stderr.write("[debug] OPENAI_API_KEY is not set\n")
        return None

    client = OpenAI()
    last_reason = None

    for max_out in budgets:
        if debug:
            plen = len(input_payload) if isinstance(input_payload, str) else sum(len(m.get("content","")) for m in input_payload)
            sys.stderr.write(
                "\n[debug] OpenAI request (responses.create)\n"
                f"  model: {model}\n"
                f"  max_output_tokens: {max_out}\n"
                f"  prompt length: {plen} chars\n"
                "  reasoning.effort: low\n"
                "  text.verbosity: low\n"
                f"  instructions: {bool(pass_instructions)}\n"
            )

        kwargs = dict(
            model=model,
            input=input_payload,
            max_output_tokens=max_out,
            reasoning={"effort": "low"},
            tool_choice="none",
            text={"verbosity": "low"},
        )
        if pass_instructions:
            kwargs["instructions"] = pass_instructions

        try:
            resp = _responses_create(client, **kwargs)
        except Exception as e:
            if debug:
                sys.stderr.write(f"[debug] responses.create exception: {e}\n")
            continue

        data = _resp_to_dict(resp) or {}
        status = data.get("status")
        reason = (data.get("incomplete_details") or {}).get("reason")
        usage = data.get("usage") or {}
        out_tok = usage.get("output_tokens", 0)
        r_tokens = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)

        if debug:
            sys.stderr.write(
                f"[debug] status={status} reason={reason} "
                f"output_tokens={out_tok} reasoning_tokens={r_tokens}\n"
            )

        text = _extract_text(resp)
        if text:
            return " ".join(text.split())

        last_reason = reason
        if not (status == "incomplete" and reason == "max_output_tokens"):
            break

        time.sleep(0.1)

    if debug and last_reason:
        sys.stderr.write(f"[debug] Gave up after escalations; last reason={last_reason}\n")
    return None

# ----------------- Summary Maintenance -----------------

def update_running_summary(
    *,
    model: str,
    budgets: List[int],
    debug: bool,
    prior_summary: str,
    last_user: str,
    last_assistant: str,
    max_chars: int,
) -> str:
    """
    Compress prior_summary + last exchange into a shorter summary ≤ max_chars.
    """
    prompt = [
        {"role": "system", "content": (
            "You maintain a terse running conversation summary for downstream turns.\n"
            f"Keep it ≤ {max_chars} characters. Use short sentences or bullets.\n"
            "Preserve key decisions, constraints, variables, IDs, URLs, and instructions.\n"
            "Drop filler, pleasantries, or redundant phrasing."
        )},
        {"role": "user", "content": (
            "Prior summary:\n"
            f"{prior_summary or '(none)'}\n\n"
            "New exchange:\n"
            f"User: {last_user}\n"
            f"Assistant: {last_assistant}\n\n"
            "Return the updated summary only."
        )}
    ]
    new_sum = call_openai_with_escalation(
        prompt,
        model=model,
        budgets=budgets[:1],  # tiny call
        debug=debug,
        pass_instructions=None,  # already in system message
    )
    if not new_sum:
        # Fallback: naive local truncation
        merged = (prior_summary + " " if prior_summary else "") + f"User asked: {last_user} | Ans: {last_assistant}"
        return (merged[:max_chars]).strip()
    # Hard cap to avoid growth if model exceeds target slightly
    if len(new_sum) > max_chars:
        new_sum = new_sum[:max_chars].rstrip()
    return new_sum

# ----------------- REPL -----------------

def run_repl(
    model: str,
    budgets: List[int],
    instructions: Optional[str],
    debug: bool,
    stream: bool,
    max_pairs: int,
    autosummary: bool,
    summary_every: int,
    summary_max_chars: int,
):
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    user_history: List[str] = []
    running_summary: str = ""
    exchanges = 0

    # Startup help
    print("oai REPL started.")
    print("Type your message. Special commands:")
    print("  /quit  → exit the session")
    print("  /reset → clear conversation history and summary\n")

    try:
        while True:
            try:
                # Two newlines before each "You:"
                user = input("\nYou: ").strip()
            except EOFError:
                print()
                break

            if not user:
                continue
            low = user.lower()
            if low in ("/quit", "/exit"):
                break
            if low == "/reset":
                user_history.clear()
                running_summary = ""
                exchanges = 0
                print("(history & summary cleared)")
                continue

            messages = build_messages_only_users(
                user_history=user_history,
                current_user=user,
                instructions=instructions,
                running_summary=running_summary,
                max_pairs=max_pairs,
            )

            if stream:
                kwargs = dict(
                    model=model,
                    input=messages,
                    max_output_tokens=budgets[-1],
                    reasoning={"effort": "low"},
                    tool_choice="none",
                    text={"verbosity": "low"},
                )
                if debug:
                    plen = sum(len(m.get("content","")) for m in messages)
                    sys.stderr.write(
                        "\n[debug] OpenAI request (responses.stream)\n"
                        f"  model: {model}\n"
                        f"  max_output_tokens: {budgets[-1]}\n"
                        f"  prompt length (chars): {plen}\n"
                    )
                assistant_chunks: List[str] = []
                try:
                    with client.responses.stream(**kwargs) as stream_obj:
                        for event in stream_obj:
                            if event.type == "response.output_text.delta":
                                chunk = event.delta
                                assistant_chunks.append(chunk)
                                print(chunk, end="", flush=True)
                        print()
                except KeyboardInterrupt:
                    print("\n(^C) stream interrupted")
                    break
                assistant_full = "".join(assistant_chunks).strip()
            else:
                assistant_full = call_openai_with_escalation(
                    messages,
                    model=model,
                    budgets=budgets,
                    debug=debug,
                    pass_instructions=None,
                ) or ""
                print(assistant_full)

            # Update local state
            user_history.append(user)
            exchanges += 1

            # Always show message before updating summary
            if autosummary and assistant_full and (exchanges % summary_every == 0 or summary_every == 1):
                print("\n(Updating conversation history and summary, please wait...)", flush=True)
                running_summary = update_running_summary(
                    model=model,
                    budgets=[min(budgets)],
                    debug=debug,
                    prior_summary=running_summary,
                    last_user=user,
                    last_assistant=assistant_full,
                    max_chars=summary_max_chars,
                )

    except KeyboardInterrupt:
        print("\n(^C) exiting REPL")



# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Prompt OpenAI (GPT-5 Mini) and print the response.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Override model (default: gpt-5-mini)")
    ap.add_argument("--budgets", default="768,1536", help="Comma-separated max_output_tokens attempts")
    ap.add_argument("--instructions",
                    default="Answer directly without asking follow-up questions. If details are missing, make reasonable assumptions and state them briefly.",
                    help="System-style instructions to steer behavior.")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--stream", action="store_true", help="Stream assistant tokens in real time")
    ap.add_argument("--history", type=int, default=DEFAULT_HISTORY_TURNS,
                    help=f"Number of prior user turns to keep (default: {DEFAULT_HISTORY_TURNS})")
    ap.add_argument("--one-shot", "--oneshot", dest="one_shot", action="store_true",
                    help="Force one-shot mode even if stdin is a TTY")
    ap.add_argument("--no-autosummary", dest="autosummary", action="store_false",
                    help="Disable running summary updates")
    ap.add_argument("--summary-every", type=int, default=DEFAULT_SUMMARY_EVERY,
                    help=f"Summarize every N exchanges (default: {DEFAULT_SUMMARY_EVERY})")
    ap.add_argument("--summary-max-chars", type=int, default=DEFAULT_SUMMARY_MAX_CHARS,
                    help=f"Maximum characters for the running summary (default: {DEFAULT_SUMMARY_MAX_CHARS})")
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x.strip().isdigit()] or [768, 1536]

    # One-shot mode if piped or explicitly requested
    if args.one_shot or not sys.stdin.isatty():
        prompt = read_prompt()
        if not prompt:
            print("No prompt provided.")
            raise SystemExit(1)

        # Build messages with empty history but include system+summary
        messages = build_messages_only_users(
            user_history=[],
            current_user=prompt,
            instructions=args.instructions,
            running_summary="",  # none in one-shot
            max_pairs=0,
        )

        if args.stream:
            client = OpenAI()
            kwargs = dict(
                model=args.model,
                input=messages,
                max_output_tokens=budgets[-1],
                reasoning={"effort": "low"},
                tool_choice="none",
                text={"verbosity": "low"},
            )
            with client.responses.stream(**kwargs) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        print(event.delta, end="", flush=True)
                print()
            return

        text = call_openai_with_escalation(
            messages,
            model=args.model,
            budgets=budgets,
            debug=args.debug,
            pass_instructions=None,  # already included in system message
        )
        if text:
            print(text)
            return
        print("[no text returned]")
        raise SystemExit(2)

    # REPL mode
    run_repl(
        model=args.model,
        budgets=budgets,
        instructions=args.instructions,
        debug=args.debug,
        stream=args.stream,
        max_pairs=args.history,
        autosummary=args.autosummary,
        summary_every=max(1, args.summary_every),
        summary_max_chars=max(200, args.summary_max_chars),
    )

if __name__ == "__main__":
    raise SystemExit(main())
