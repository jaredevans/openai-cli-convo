# oai.py

`oai.py` is a python script that provides a command-line interface (CLI) for interacting with *reasoning* GPTs via the OpenAI API. It functions as a REPL (Read-Eval-Print Loop), allowing for continuous conversation with an AI model.

## Features

- **REPL Interface**: Engages in a continuous conversation until you exit.
- **Server-Side Conversations**: Keeps track of the conversation context on the server-side.
- **Command-Line Flags**: Customize the model, budget, and other settings.
- **Streaming Support**: Can stream responses from the API for faster interaction.
- **One-Shot Mode**: Used for single query then exit immediately.

<img src="https://i.imgur.com/dZN2JSf.png">

## Usage

To start the REPL, run the script from your terminal:

- Python 3
- An OpenAI API key set as an environment variable: `export OPENAI_API_KEY='your-key'`

### Installation

1.  Clone the repository 
2.  Install the required dependencies:

# uv (preferred)
```
uv sync
uv run python oai.py --stream
```
# pip
```
pip install -r requirements.txt
python oai.py --stream
```

# Conversations

You can then type your prompts and press Enter. To exit, type `/quit` or use `Ctrl-C`.

You will be in a conversation with looping prompting with memorized history of previous prompts and responses.

`/reset` to clear out the history and start a new conversation.

### Special Commands

- `/quit`: Exit the REPL.
- `/reset`: Start a new conversation, clearing the previous context.

### One-Shot Mode

For single interactions, you can use the pipe input to the script.

```bash
echo "What is the capital of France?" | uv run python oai.py
The capital of France is Paris.

% uv run python oai.py --one-shot
Enter your prompt: What is the capital of France?
The capital of France is Paris.
```

## Options

The script supports several command-line arguments to customize its behavior:

- `--model NAME`: Specify the reasoning model to use (default: `gpt-5-mini`).
- `--system TEXT`: Provide system instructions for the AI. (default: "Answer directly and concisely.").
- `--budgets "A,B,..."`: A comma-separated list of response token budgets to try (default: `"800,1600"`).
- `--effort CHOICE`: The effort level for the model, from "low", "medium", or "high" (default: "medium").
- `--stream`: Enable streaming for the response.
- `--one-shot`: Use the script for a single interaction.

## Requirements

- Python 3
- `openai` library

You can install the required library using pip:

```bash
uv sync 
  -or- 
pip install openai
```

You also need to have your OpenAI API key set as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

---

## endpoint-oai.py

`endpoint-oai.py` is a Python script that provides a REPL (Read-Eval-Print Loop) for chatting with any OpenAI-compatible API endpoint, such as [LM Studio](https://lmstudio.ai/). It's designed for continuous conversation with session-only memory management, automatically compressing old messages into a running summary to avoid exceeding the context window. It also includes a `/check` command to verify the accuracy of the last AI response using high-end OpenAI API. The `/check` command is useful when you're prompting smaller, less capable AI models.

### Features

- **OpenAI-Compatible Endpoint**: Connects to any service that provides a `/v1/chat/completions` endpoint.
- **Automatic Context Compression**: When the conversation gets long, it automatically summarizes the oldest parts of the dialogue, allowing for very long conversations.
- **Response Verification**: Use the `/check` command to get a second opinion on the AI's last answer from an OpenAI model (e.g., `gpt-5-mini`).
- **REPL Interface**: For continuous conversation.
- **Configurable**: Key parameters can be configured via environment variables.

<img src ="https://i.imgur.com/GS7mvRr.png">

### Usage

To start the REPL, run the script from your terminal:

```bash
python endpoint-oai.py
```

### Special Commands

- `/quit` or `/exit`: Exit the REPL.
- `/reset`: Clear the conversation history.
- `/model <name>`: Switch to a different model.
- `/system <text>`: Change the base system prompt.
- `/summary`: View the current compressed summary of the conversation.
- `/summarize`: Force an update of the conversation summary.
- `/check`: Verify the last AI response for accuracy using the OpenAI API.
- `/help`: Show the list of available commands.

### Configuration

The script can be configured using the following environment variables:

**For the LM Studio / local endpoint:**

- `LMSTUDIO_BASE_URL`: The base URL of the endpoint (default: `http://127.0.0.1:11435/v1`).
- `LMSTUDIO_MODEL`: The default model name to use.
- `LMSTUDIO_API_KEY`: The API key if required (default: `not-needed`).
- `LMSTUDIO_KEEP_TURNS`: Number of recent turns to keep in full detail (default: `8`).
- `LMSTUDIO_MAX_CONTEXT_CHARS`: Character limit to trigger compression (default: `12000`).
- `LMSTUDIO_SUMMARY_WORDS`: Target word count for the summary (default: `120`).

**For the `/check` command:**

- `OPENAI_API_KEY`: Your OpenAI API key (required to use `/check`).
- `OAI_CHECK_MODEL`: The model to use for verification (default: `gpt-5-mini`).
- `OAI_CHECK_DEBUG`: Set to `1`, `true`, or `yes` to print debug information.
