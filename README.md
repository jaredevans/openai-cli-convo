# oai.py

`oai.py` is a Python script that provides a command-line interface (CLI) for interacting with the OpenAI API. It functions as a REPL (Read-Eval-Print Loop), allowing for continuous conversation with an AI model.

## Features

- **REPL Interface**: Engages in a continuous conversation until you exit.
- **Server-Side Conversations**: Keeps track of the conversation context on the server-side.
- **Command-Line Flags**: Customize the model, budget, and other settings.
- **Streaming Support**: Can stream responses from the API for faster interaction.
- **One-Shot Mode**: Can be used for single queries, especially useful when piping input.

<img src="https://i.imgur.com/dZN2JSf.png">

<img src="https://i.imgur.com/F3NOq2a.png">

## Usage

To start the REPL, run the script from your terminal:

<<<<<<< HEAD
- Python 3
- An OpenAI API key set as an environment variable: `export OPENAI_API_KEY='your-key'`

### Installation

1.  Clone the repository or download the `oai.py` script.
2.  Install the required dependencies:

# uv (preferred)
```
uv sync
uv run python oai.py --stream
```
# pip
```
pip install -r requirements.txt
=======
```bash
>>>>>>> e16e21f (server-side conversations to keep track of previous messages in convo.)
python oai.py
```

You can then type your prompts and press Enter. To exit, type `/quit` or use `Ctrl-C`.

### Special Commands

- `/quit`: Exit the REPL.
- `/reset`: Start a new conversation, clearing the previous context.

### One-Shot Mode

For single interactions, you can use the `--one-shot` flag or pipe input to the script.

**Using the flag:**

```bash
uv run python oai.py --one-shot "What is the capital of France?"
```

**Piping input:**

```bash
echo "What is the capital of France?" | uv run python oai.py
```

## Options

The script supports several command-line arguments to customize its behavior:

- `--model NAME`: Specify the model to use (default: `gpt-5-mini`).
- `--budgets "A,B,..."`: A comma-separated list of response token budgets to try (default: `"768,1536"`).
- `--instructions TEXT`: Provide system instructions for the AI.
- `--stream`: Enable streaming for the response.
- `--one-shot` or `--oneshot`: Use the script for a single interaction.
- `--debug`: Enable debug output to stderr.

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
