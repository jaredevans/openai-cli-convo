#!/usr/bin/env python3
"""
LM Studio Chat REPL (session memory with auto-compression + on-demand /summarize)

- Uses OpenAI-compatible /v1 endpoint (For instance: LM Studio on http://127.0.0.1:11435).
- Keeps recent turns verbatim; folds older context into a compact system summary.
- Session-only: no files written or read.

Commands:
  /quit or /exit      → exit
  /reset              → clear history (keeps base system)
  /model <name>       → switch model (default from LMSTUDIO_MODEL or gemma-3-12b)
  /system <text>      → set/replace the base system prompt
  /summary            → print current compressed summary (if any)
  /summarize          → force-create/update the compressed summary now
  /help               → show commands
"""

import os
import sys
from typing import List, Dict, Tuple

from openai import OpenAI

# --- Config (env overridable) ---
BASE_URL       = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:11435/v1")
DEFAULT_MODEL  = os.environ.get("LMSTUDIO_MODEL", "google/gemma-3-12b")
API_KEY        = os.environ.get("LMSTUDIO_API_KEY", "not-needed")  # required by client
# How many *turns* to keep verbatim (each turn = user+assistant)
KEEP_TURNS     = int(os.environ.get("LMSTUDIO_KEEP_TURNS", "8"))
# Trigger compression when combined char length exceeds this:
MAX_CONTEXT_CHARS = int(os.environ.get("LMSTUDIO_MAX_CONTEXT_CHARS", "12000"))
# Target size for the summary (soft goal; summarizer keeps it tight)
SUMMARY_TARGET_WORDS = int(os.environ.get("LMSTUDIO_SUMMARY_WORDS", "120"))

BASE_SYSTEM = (
    "You are a concise, helpful assistant. Be direct, accurate, and pragmatic. "
    "Prefer clear steps and minimal fluff."
)

def est_total_chars(msgs: List[Dict]) -> int:
    return sum(len(m.get("content", "")) for m in msgs)

def build_summary_prompt(existing_summary: str, transcript_chunk: List[Dict]) -> List[Dict]:
    # Turn transcript chunk into a simple plain-text log
    lines = []
    for m in transcript_chunk:
        role = m.get("role", "user")
        content = m.get("content", "").strip()
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")
    chunk_text = "\n".join(lines)

    sys_msg = {
        "role": "system",
        "content": (
            "You are a compression assistant. Given a running chat summary and a new transcript chunk, "
            "produce a *tight* updated summary that preserves:\n"
            "- user goals, constraints, preferences\n"
            "- key facts/definitions/IDs/examples\n"
            "- decisions made and rationale\n"
            "- open questions / next steps\n\n"
            f"Limit to ~{SUMMARY_TARGET_WORDS} words, bullet-like prose if helpful. "
            "No pleasantries, no filler, avoid repetition, keep technical detail that affects answers. "
            "Do NOT include code unless essential."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"CURRENT SUMMARY (may be empty):\n{existing_summary or '(none)'}\n\n"
            f"NEW TRANSCRIPT CHUNK:\n{chunk_text}\n\n"
            "Return only the UPDATED SUMMARY."
        ),
    }
    return [sys_msg, user_msg]

def summarize_now(client: OpenAI, model: str, summary_system_msg: Dict, messages: List[Dict]) -> Dict:
    """Force-create/update the compressed summary from all user/assistant turns so far."""
    existing = summary_system_msg.get("content", "") if summary_system_msg else ""
    convo = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not convo and not existing:
        return summary_system_msg  # nothing to summarize
    smsgs = build_summary_prompt(existing, convo)
    try:
        sresp = client.chat.completions.create(model=model, messages=smsgs, temperature=0.2)
        updated = (sresp.choices[0].message.content or "").strip()
        if not summary_system_msg:
            summary_system_msg = {"role": "system", "content": ""}
        summary_system_msg["content"] = f"[Dialogue Summary]\n{updated}"
    except Exception as e:
        print(f"[summarize error: {e}]")
    return summary_system_msg

def compress_if_needed(
    client: OpenAI,
    model: str,
    base_system_msg: Dict,
    summary_system_msg: Dict,
    messages: List[Dict],
) -> Tuple[Dict, Dict, List[Dict], str]:
    """
    If total context is large, fold older turns into the summary system message.
    Returns possibly updated (base_system_msg, summary_system_msg, messages, summary_text).
    """
    total_chars = est_total_chars([base_system_msg] + ([summary_system_msg] if summary_system_msg else []) + messages)
    if total_chars <= MAX_CONTEXT_CHARS and len(messages) <= KEEP_TURNS * 2 + 2:
        # No compression needed
        return base_system_msg, summary_system_msg, messages, (summary_system_msg.get("content", "") if summary_system_msg else "")

    # Identify chunk to fold: everything except the most recent KEEP_TURNS*2 role msgs
    non_system = [m for m in messages if m.get("role") in ("user", "assistant")]
    tail = non_system[-KEEP_TURNS * 2 :] if KEEP_TURNS > 0 else []
    head_len = len(non_system) - len(tail)
    head = non_system[:head_len] if head_len > 0 else []

    if not head:
        # Nothing to fold; still too big? Then trim tail conservatively.
        trimmed = tail[-KEEP_TURNS * 2 :]
        return base_system_msg, summary_system_msg, trimmed, (summary_system_msg.get("content", "") if summary_system_msg else "")

    existing_summary = summary_system_msg.get("content", "") if summary_system_msg else ""
    # Ask model to update the summary
    try:
        smsgs = build_summary_prompt(existing_summary, head)
        sresp = client.chat.completions.create(model=model, messages=smsgs, temperature=0.2)
        updated_summary = (sresp.choices[0].message.content or "").strip()
        if not summary_system_msg:
            summary_system_msg = {"role": "system", "content": ""}
        summary_system_msg["content"] = f"[Dialogue Summary]\n{updated_summary}"
    except Exception:
        pass  # keep existing summary if update fails

    # Rebuild working messages = recent tail only (plus any non user/assistant messages if present)
    new_messages = []
    others = [m for m in messages if m.get("role") not in ("user", "assistant")]
    new_messages.extend(others)
    new_messages.extend(tail)

    return base_system_msg, summary_system_msg, new_messages, summary_system_msg.get("content", "") if summary_system_msg else ""

def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    current_model = DEFAULT_MODEL
    base_system_msg = {"role": "system", "content": BASE_SYSTEM}
    summary_system_msg = None  # {"role": "system", "content": "[Dialogue Summary] ..."}
    messages: List[Dict] = []

    print(f"LM Studio Chat (model: {current_model}) — session memory enabled")
    print("Type /help for commands. Ctrl+C or /quit to exit.\n")

    def show_help():
        print(
            "Commands:\n"
            "  /quit or /exit      Exit\n"
            "  /reset              Clear history (keeps base system)\n"
            "  /model <name>       Switch model\n"
            "  /system <text>      Set/replace base system prompt\n"
            "  /summary            Show current compressed summary\n"
            "  /summarize          Force-create/update the compressed summary now\n"
            "  /help               Show this help\n"
        )

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if not user_input:
            continue

        # Slash-commands
        if user_input.startswith("/"):
            cmd, *rest = user_input.split(maxsplit=1)
            arg = rest[0] if rest else ""

            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/help":
                show_help()
                continue
            elif cmd == "/reset":
                messages = []
                summary_system_msg = None
                print("[history cleared]")
                continue
            elif cmd == "/model":
                if arg:
                    current_model = arg
                    print(f"[model set to: {current_model}]")
                else:
                    print(f"[current model: {current_model}]")
                continue
            elif cmd == "/system":
                if arg:
                    base_system_msg["content"] = arg
                    print("[base system prompt updated]")
                else:
                    print("[usage] /system <text>")
                continue
            elif cmd == "/summary":
                if summary_system_msg and summary_system_msg.get("content"):
                    print(summary_system_msg["content"])
                else:
                    print("[no summary yet]")
                continue
            elif cmd == "/summarize":
                summary_system_msg = summarize_now(client, current_model, summary_system_msg, messages)
                if summary_system_msg and summary_system_msg.get("content"):
                    print(summary_system_msg["content"])
                else:
                    print("[no summary yet]")
                continue
            else:
                print("[unknown command; /help]")
                continue

        # Normal chat turn
        messages.append({"role": "user", "content": user_input})

        # Compress if needed before sending
        base_system_msg, summary_system_msg, messages, _ = compress_if_needed(
            client, current_model, base_system_msg, summary_system_msg, messages
        )

        try:
            resp = client.chat.completions.create(
                model=current_model,
                messages=[base_system_msg] + ([summary_system_msg] if summary_system_msg else []) + messages,
                temperature=0.7,
            )
            reply = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            reply = f"[error from server: {e}]"

        print(f"AI: {reply}\n")
        messages.append({"role": "assistant", "content": reply})

        # Optional: opportunistic compress after each turn if we’ve grown too large
        base_system_msg, summary_system_msg, messages, _ = compress_if_needed(
            client, current_model, base_system_msg, summary_system_msg, messages
        )

if __name__ == "__main__":
    main()
