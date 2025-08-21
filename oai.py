#!/usr/bin/env python3
"""
oai.py — REPL using Responses API with server-side convo via previous_response_id

Behavior:
- REPL until /quit or Ctrl-C (one-shot if piped or --one-shot).
- Server-side conversation only: tracked by previous_response_id.
- Two blank lines before each "You:".
- No "Assistant:" prefix.
- /reset starts fresh conversation (clears previous_response_id).

Flags:
  --model NAME
  --budgets "A,B,..." (default "768,1536")
  --instructions TEXT
  --stream
  --one-shot/--oneshot
  --debug
"""

import os, sys, time, argparse
from typing import Optional, List, Union, Tuple
from openai import OpenAI

DEFAULT_MODEL = "gpt-5-mini"

# ----------------- Helpers -----------------

def _resp_to_dict(resp):
    for attr in ("to_dict", "model_dump"):
        if hasattr(resp, attr):
            try:
                return getattr(resp, attr)()
            except Exception:
                pass
    return {}

def _extract_text(resp) -> Optional[str]:
    # Fast path (new SDKs)
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # Responses API shapes
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

    # Sometimes present as {"text":{"value": "..."}}
    tnode = data.get("text")
    if isinstance(tnode, dict):
        tval = tnode.get("value")
        if isinstance(tval, str) and tval.strip():
            return tval.strip()

    # Very old fallback (chat.completions-like)
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = (((choices[0] or {}).get("message") or {}).get("content") or "")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    return None

def _responses_create(client: OpenAI, **kwargs):
    """Call responses.create; if 'text' unsupported by this SDK, retry without it."""
    try:
        return client.responses.create(**kwargs)
    except TypeError:
        if "text" in kwargs:
            kwargs2 = dict(kwargs)
            kwargs2.pop("text", None)
            return client.responses.create(**kwargs2)
        raise

def read_prompt():
    if not sys.stdin.isatty():
        txt = sys.stdin.read().strip()
        if txt:
            return txt
    try:
        return input("Enter your prompt: ").strip()
    except EOFError:
        return ""

# ----------------- Core request -----------------

def call_and_handle(
    client: OpenAI,
    model: str,
    budgets: List[int],
    debug: bool,
    messages_or_text: Union[str, list],
    prev_id: Optional[str],
    stream: bool,
) -> Tuple[str, Optional[str]]:
    """
    Makes a request (stream or non-stream), returns (assistant_text, new_response_id).
    Uses previous_response_id to continue server-side conversation.
    """
    last_error = None

    for max_out in budgets:
        # Build kwargs once per attempt
        kwargs = dict(
            model=model,
            input=messages_or_text,
            max_output_tokens=max_out,
            reasoning={"effort": "low"},
            tool_choice="none",
            text={"verbosity": "low"},
        )
        if prev_id:
            kwargs["previous_response_id"] = prev_id

        if debug:
            plen = len(messages_or_text) if isinstance(messages_or_text, str) \
                   else sum(len(m.get("content","")) for m in messages_or_text)
            sys.stderr.write(
                "\n[debug] request (responses"
                f"{'.stream' if stream else '.create'})\n"
                f"  model: {model}\n"
                f"  max_output_tokens: {max_out}\n"
                f"  prompt length (chars): {plen}\n"
                f"  previous_response_id: {prev_id}\n"
            )

        try:
            if stream:
                # Use the streaming context manager so we can capture the final response id
                new_id = None
                chunks: List[str] = []
                with client.responses.stream(**kwargs) as stream_obj:
                    for event in stream_obj:
                        # Print tokens live
                        if event.type == "response.output_text.delta":
                            print(event.delta, end="", flush=True)
                            chunks.append(event.delta)
                        # Capture id at the end
                        elif event.type == "response.completed":
                            # event.response.id should be present
                            try:
                                new_id = event.response.id  # SDK object
                            except Exception:
                                # Fallback if different shape
                                ev = getattr(event, "response", None)
                                if isinstance(ev, dict):
                                    new_id = ev.get("id")

                    # Ensure a newline after stream
                    print()

                # If not captured above, try last-resort extraction
                if new_id is None:
                    try:
                        final = stream_obj.get_final_response()
                        new_id = getattr(final, "id", None) or (_resp_to_dict(final) or {}).get("id")
                    except Exception:
                        pass

                return ("".join(chunks).strip(), new_id)

            else:
                resp_obj = _responses_create(client, **kwargs)
                new_id = getattr(resp_obj, "id", None) or (_resp_to_dict(resp_obj) or {}).get("id")
                text = _extract_text(resp_obj) or ""
                # Detect incomplete due to max tokens and escalate
                data = _resp_to_dict(resp_obj) or {}
                status = data.get("status")
                reason = (data.get("incomplete_details") or {}).get("reason")
                if status == "incomplete" and reason == "max_output_tokens":
                    # Escalate to next budget
                    last_error = "max_output_tokens"
                    time.sleep(0.05)
                    continue
                return (text, new_id)

        except Exception as e:
            last_error = str(e)
            if debug:
                sys.stderr.write(f"[debug] exception: {e}\n")
            time.sleep(0.1)

    if debug and last_error:
        sys.stderr.write(f"[debug] giving up; last error: {last_error}\n")
    return ("", None)

# ----------------- REPL -----------------

def run_repl(model, budgets, instructions, debug, stream):
    if not os.getenv("OPENAI_API_KEY"):
        print("API key not found.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    prev_id: Optional[str] = None

    # Startup help
    print("oai REPL started.")
    print("Special commands:")
    print("  /quit  → exit")
    print("  /reset → fresh conversation (server-side)\n")

    try:
        while True:
            # Two blank lines before prompt
            user = input("\n\nYou: ").strip()
            if not user:
                continue
            low = user.lower()
            if low in ("/quit", "/exit"):
                break
            if low == "/reset":
                prev_id = None
                print("(conversation reset)")
                continue

            # Build minimal messages for this turn
            messages = []
            if instructions:
                messages.append({"role": "system", "content": instructions})
            messages.append({"role": "user", "content": user})

            # Send
            assistant_text, new_id = call_and_handle(
                client=client,
                model=model,
                budgets=budgets,
                debug=debug,
                messages_or_text=messages,
                prev_id=prev_id,
                stream=stream,
            )

            # Print reply (no "Assistant:" prefix)
            if not stream:
               print(assistant_text)

            # Chain the conversation server-side
            if new_id:
                prev_id = new_id

    except KeyboardInterrupt:
        print("\n(^C) exiting session")

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--budgets", default="768,1536")
    ap.add_argument("--instructions",
                    default="Answer directly without asking follow-up questions. If details are missing, make reasonable assumptions and state them briefly.")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--one-shot", "--oneshot", dest="one_shot", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",") if x.strip().isdigit()] or [768, 1536]

    # One-shot mode (piped input or explicit flag)
    if args.one_shot or not sys.stdin.isatty():
        prompt = read_prompt()
        if not prompt:
            print("No prompt given.")
            sys.exit(1)
        client = OpenAI()
        prev_id = None
        messages = []
        if args.instructions:
            messages.append({"role": "system", "content": args.instructions})
        messages.append({"role": "user", "content": prompt})
        text, _ = call_and_handle(
            client=client,
            model=args.model,
            budgets=budgets,
            debug=args.debug,
            messages_or_text=messages,
            prev_id=prev_id,
            stream=args.stream,
        )

        if not args.stream:
            print(text)

        return

    # REPL mode
    run_repl(args.model, budgets, args.instructions, args.debug, args.stream)

if __name__ == "__main__":
    main()
