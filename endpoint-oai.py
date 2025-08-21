#!/usr/bin/env python3
"""
LM Studio Chat REPL (session memory with auto-compression + on-demand /summarize + /check)

Commands:
  /quit or /exit      → exit
  /reset              → clear history (keeps base system)
  /model <name>       → switch model (default from LMSTUDIO_MODEL or google/gemma-3-12b)
  /system <text>      → set/replace the base system prompt
  /summary            → print current compressed summary (if any)
  /summarize          → force-create/update the compressed summary now
  /check              → verify last answer using OpenAI Responses API (designed for gpt-5-mini)
  /help               → show commands

Env:
  LMSTUDIO_BASE_URL          (default: http://127.0.0.1:11435/v1)
  LMSTUDIO_MODEL             (default: google/gemma-3-12b)
  LMSTUDIO_API_KEY           (dummy for LM Studio; default: "not-needed")
  LMSTUDIO_KEEP_TURNS        (default: 8)
  LMSTUDIO_MAX_CONTEXT_CHARS (default: 12000)
  LMSTUDIO_SUMMARY_WORDS     (default: 120)

  OPENAI_API_KEY             (required for /check)
  OAI_CHECK_MODEL            (default: gpt-5-mini)
  OAI_CHECK_DEBUG            (1/true/yes to print raw responses on parse miss)
"""

import os
import sys
from typing import List, Dict, Tuple, Optional, Any

from openai import OpenAI

# --- Config (env overridable) ---
BASE_URL             = os.environ.get("LMSTUDIO_BASE_URL", "http://127.0.0.1:11435/v1")
DEFAULT_MODEL        = os.environ.get("LMSTUDIO_MODEL", "google/gemma-3-12b")
API_KEY              = os.environ.get("LMSTUDIO_API_KEY", "not-needed")  # required by LM Studio client
KEEP_TURNS           = int(os.environ.get("LMSTUDIO_KEEP_TURNS", "8"))
MAX_CONTEXT_CHARS    = int(os.environ.get("LMSTUDIO_MAX_CONTEXT_CHARS", "12000"))
SUMMARY_TARGET_WORDS = int(os.environ.get("LMSTUDIO_SUMMARY_WORDS", "120"))
CHECK_MODEL          = os.environ.get("OAI_CHECK_MODEL", "gpt-5-mini")
CHECK_DEBUG          = os.environ.get("OAI_CHECK_DEBUG", "").lower() in ("1", "true", "yes")

BASE_SYSTEM = (
    "You are a concise, helpful assistant. Be direct, accurate, and pragmatic. "
    "Prefer clear steps and minimal fluff."
)

# =========================
# Helpers for summaries/etc
# =========================

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
        sresp = client.chat.completions.create(model=model, messages=smsgs)
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
        sresp = client.chat.completions.create(model=model, messages=smsgs)
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

# ==========================================
# /check — Responses API (like your oai.py)
# ==========================================

def _resp_to_dict(resp: Any) -> Dict:
    for attr in ("to_dict", "model_dump"):
        if hasattr(resp, attr):
            try:
                return getattr(resp, attr)()
            except Exception:
                pass
    return {}

def _extract_responses_text(resp: Any) -> Optional[str]:
    """
    Extract text from Responses API objects across shapes.
    """
    # Preferred modern path
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    d = _resp_to_dict(resp) or {}
    outputs = d.get("output") or d.get("outputs") or []
    for item in outputs:
        for part in item.get("content", []) or []:
            t = part.get("text")
            if isinstance(t, str) and t.strip():
                return t.strip()
            if part.get("type") == "message":
                for mc in part.get("content", []) or []:
                    if mc.get("type") == "text":
                        t2 = (mc.get("text") or "").strip()
                        if t2:
                            return t2

    # Very old chat-like fallback
    choices = d.get("choices")
    if isinstance(choices, list) and choices:
        msg = (((choices[0] or {}).get("message") or {}).get("content") or "")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    return None

def check_accuracy(last_user: str, last_assistant: str) -> None:
    """
    Accuracy check tuned for gpt-5-mini:
      - Large max_output_tokens so text can follow reasoning
      - Cap hidden reasoning via max_reasoning_tokens
      - Escalate budgets on 'incomplete/max_output_tokens'
      - Graceful fallbacks if SDK rejects optional params
    Env:
      OPENAI_API_KEY              required
      OAI_CHECK_MODEL             default gpt-5-mini
      OAI_CHECK_BUDGETS           e.g. "256,512,1024" (default used below)
      OAI_CHECK_REASONING_CAP     integer cap (default 64)
      OAI_CHECK_DEBUG             1/true/yes for raw dumps on parse miss
    """
    import os
    from typing import Any, Dict, Optional
    from openai import OpenAI

    oc = OpenAI()
    model_name = os.environ.get("OAI_CHECK_MODEL", "gpt-5-mini")
    debug = os.environ.get("OAI_CHECK_DEBUG", "").lower() in ("1", "true", "yes")

    # Budgets: plenty of headroom so there's leftover for text even after reasoning
    budgets_env = os.environ.get("OAI_CHECK_BUDGETS", "256,512,1024")
    budgets = [int(x) for x in budgets_env.split(",") if x.strip().isdigit()] or [256, 512, 1024]

    # Cap the hidden chain so it can’t eat the whole budget
    try:
        reasoning_cap = int(os.environ.get("OAI_CHECK_REASONING_CAP", "64"))
    except Exception:
        reasoning_cap = 64

    sys_prompt = (
        "You are an evaluator. Given the USER's prompt and the ASSISTANT's reply, "
        "decide if the reply is accurate and correct. Respond with EXACTLY:\n"
        "Yes — <short one-line justification>\n"
        "or\n"
        "No — <short one-line justification>\n"
        "Requirements: Start with 'Yes' or 'No'. Keep total under ~20 tokens."
    )
    user_block = f"USER PROMPT:\n{last_user}\n\nASSISTANT REPLY:\n{last_assistant}\n\nWas it accurate?"

    # ---- tiny local fallbacks if your file doesn't already define these helpers ----
    def _to_dict(obj: Any) -> Dict:
        for attr in ("to_dict", "model_dump", "dict"):
            if hasattr(obj, attr):
                try:
                    return getattr(obj, attr)()
                except Exception:
                    pass
        return getattr(obj, "__dict__", {}) or {}

    def _extract_responses_text_local(resp: Any) -> Optional[str]:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        d = _to_dict(resp) or {}
        outputs = d.get("output") or d.get("outputs") or []
        for item in outputs:
            for part in item.get("content", []) or []:
                t = part.get("text")
                if isinstance(t, str) and t.strip():
                    return t.strip()
                if part.get("type") == "message":
                    for mc in part.get("content", []) or []:
                        if mc.get("type") == "text":
                            t2 = (mc.get("text") or "").strip()
                            if t2:
                                return t2
        ch = d.get("choices")
        if isinstance(ch, list) and ch:
            msg = (((ch[0] or {}).get("message") or {}).get("content") or "")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        return None

    _extract_responses_text_fn = globals().get("_extract_responses_text", _extract_responses_text_local)

    # -------------------- main loop: escalate budgets --------------------
    for max_out in budgets:
        try:
            # Primary attempt: include both large output and reasoning cap
            kwargs = dict(
                model=model_name,
                input=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_block},
                ],
                tool_choice="none",
                max_output_tokens=max_out,
                max_reasoning_tokens=reasoning_cap,   # <- cap the hidden chain
                text={"verbosity": "low"},            # helpful when supported
            )

            # Some SDKs/models may reject optional params; strip and retry in order
            try:
                resp = oc.responses.create(**kwargs)
            except TypeError:
                # remove 'text'
                kwargs2 = dict(kwargs)
                kwargs2.pop("text", None)
                try:
                    resp = oc.responses.create(**kwargs2)
                except TypeError:
                    # remove reasoning cap too
                    kwargs3 = dict(kwargs2)
                    kwargs3.pop("max_reasoning_tokens", None)
                    resp = oc.responses.create(**kwargs3)

            d = _to_dict(resp) or {}
            out = _extract_responses_text_fn(resp) or ""

            if out.strip():
                print(f"[Accuracy Check] {out}")
                return

            status = d.get("status")
            reason = (d.get("incomplete_details") or {}).get("reason")
            if status == "incomplete" and reason == "max_output_tokens":
                if debug:
                    from pprint import pprint
                    print(f"[/check debug] incomplete due to max_output_tokens at {max_out} (cap={reasoning_cap})")
                    pprint(d)
                # escalate to next budget
                continue

            # Not incomplete but still empty → nothing to gain by retrying further
            if debug:
                from pprint import pprint
                print("[/check debug] empty text (not incomplete); raw follows:")
                pprint(d)
            break

        except Exception as e:
            if debug:
                print(f"[/check debug] responses attempt failed at out={max_out}: {e}")

    print("[Accuracy Check] (empty)")


# ================
# Main LM Studio UI
# ================

def main():
    # LM Studio-compatible client
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
            "  /check              Verify last answer with OpenAI API (Responses)\n"
            "  /help               Show this help\n"
        )

    while True:
        try:
            user_input = input("\nYou: ").strip()
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
            elif cmd == "/check":
                # Find the last assistant reply and its preceding user prompt
                last_user = None
                last_assistant = None
                for m in reversed(messages):
                    if not last_assistant and m.get("role") == "assistant":
                        last_assistant = m.get("content")
                    elif last_assistant and m.get("role") == "user":
                        last_user = m.get("content")
                        break
                if last_user and last_assistant:
                    check_accuracy(last_user, last_assistant)
                else:
                    print("[/check] No recent Q/A pair found.")
                continue
            else:
                print("[unknown command; /help]")
                continue

        # Normal chat turn (LM Studio)
        messages.append({"role": "user", "content": user_input})

        # Compress if needed before sending
        base_system_msg, summary_system_msg, messages, _ = compress_if_needed(
            client, current_model, base_system_msg, summary_system_msg, messages
        )

        try:
            resp = client.chat.completions.create(
                model=current_model,
                messages=[base_system_msg] + ([summary_system_msg] if summary_system_msg else []) + messages,
            )
            reply = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            reply = f"[error from server: {e}]"

        print(f"AI: {reply}\n")
        messages.append({"role": "assistant", "content": reply})

        # Optional: compress after each turn if we’ve grown too large
        base_system_msg, summary_system_msg, messages, _ = compress_if_needed(
            client, current_model, base_system_msg, summary_system_msg, messages
        )

if __name__ == "__main__":
    main()
