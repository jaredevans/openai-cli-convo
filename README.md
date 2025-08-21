# oai.py - OpenAI Command-Line Interface

A Python script for interacting with the OpenAI API from the command line. It supports both interactive (REPL) and one-shot modes.

## Features

- **Interactive REPL:** Engage in a conversation with the model.
- **One-Shot Mode:** Pipe input directly to the script for a single response.
- **Conversation History:** Maintains a history of user turns to provide context.
- **Running Summary:** Keeps a compressed summary of the conversation to stay within token limits.
- **Streaming:** Get real-time responses from the model.
- **Token Escalation:** Automatically increases the `max_output_tokens` on subsequent retries if the previous attempt failed due to token limits.

## Usage

### Prerequisites

- Python 3
- An OpenAI API key set as an environment variable: `export OPENAI_API_KEY='your-key'`

### Installation

1.  Clone the repository or download the `oai.py` script.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
# uv (preferred)
```
uv sync
uv run python oai.py --stream
```
# pip
```
pip install -r requirements.txt
python oai.py
```

### Interactive (REPL) Mode

To start the REPL, run the script without any arguments:

```bash
uv run python oai.py --stream
```

You can then type your prompts and get responses. The following commands are available:

- `/quit`: Exit the session.
- `/reset`: Clear the conversation history and summary.

### One-Shot Mode

You can pipe content to the script or use the `--one-shot` flag for a single interaction.

```bash
echo "What is the capital of France?" | uv run python oai.py
```

```bash
uv run python oai.py --one-shot "What is the capital of France?"
```

## Options

| Option                | Description                                                                      | Default                                                               |
| --------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `--model`             | The model to use for the conversation.                                           | `gpt-5-mini`                                                          |
| `--budgets`           | Comma-separated `max_output_tokens` attempts.                                    | `768,1536`                                                            |
| `--instructions`      | System-style instructions to steer the model's behavior.                         | "Answer directly without asking follow-up questions..."               |
| `--debug`             | Enable debug mode to see more verbose output.                                    | `False`                                                               |
| `--stream`            | Stream assistant tokens in real time.                                            | `False`                                                               |
| `--history`           | Number of prior user turns to keep in the history.                               | `10`                                                                  |
| `--one-shot`          | Force one-shot mode even if stdin is a TTY.                                      | `False`                                                               |
| `--no-autosummary`    | Disable running summary updates.                                                 | `False`                                                               |
| `--summary-every`     | Summarize every N exchanges.                                                     | `1`                                                                   |
| `--summary-max-chars` | Maximum characters for the running summary.                                      | `1200`                                                                |
