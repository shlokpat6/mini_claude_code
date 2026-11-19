
import json
import os
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

MODEL = "gpt-5-mini"

SYSTEM_PROMPT = """You are a coding agent running in the user's terminal.
You can list files, read files, write files, and run shell commands.
Use your tools to complete the user's task, then briefly summarize what you did.
The working directory is the folder the user launched you from."""

# Define the Python tools that I want the model to have access to

def list_files(path="."):
    entries = []

    for entry in os.scandir(path):
        entries.append(entry.name + ("/" if entry.is_dir() else ""))

    return "\n".join(sorted(entries)) or "(empty directory)"


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Saved {path} ({len(content)} characters)"


def run_command(command):
    answer = input(f"  Run '{command}'? [y/N] ")

    if answer.strip().lower() != "y":
        return "The user declined to run this command."

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=120
    )

    output = (result.stdout + result.stderr).strip()

    return output or f"(no output, exit code {result.returncode})"

# Create a list of available tools

TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}

# Creat the tool schemas

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "list_files",
        "description": "List the files in a directory. Folders end with /.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list, e.g. '.'"
                },
            },
            "required": [],
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a text file and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to read"
                },
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "write_file",
        "description": "Create or overwrite a text file with the given content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Full contents of the file"
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "type": "function",
        "name": "run_command",
        "description": "Run a shell command and return its output. The user approves it first.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run"
                },
            },
            "required": ["command"],
        },
    },
]

# Execute the requested tools

def run_tool(tool_call):

    name = tool_call.name
    args = json.loads(tool_call.arguments)

    print(f"  tool: {name}({args})")

    try:
        return str(TOOLS[name](**args))

    except Exception as error:
        return f"Error: {error}"

# Agent loop

def run_agent(input_items):

    while True:

        response = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=input_items,
            tools=TOOL_SCHEMAS,
        )

        # Save the model's output to conversation history
        input_items += response.output

        # Find all function calls
        tool_calls = [
            item for item in response.output
            if item.type == "function_call"
        ]

        # No tool calls means the model has finished
        if not tool_calls:
            return response.output_text

        # Execute each requested tool
        for tool_call in tool_calls:

            result = run_tool(tool_call)

            # Return the tool result to the model
            input_items.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": result,
            })

# Main terminal interface

def main():

    input_items = []

    print("Mini agent ready. Type 'exit' to quit.")

    while True:

        user_input = input("\nYou: ")

        if user_input.strip().lower() in ("exit", "quit"):
            break

        input_items.append({
            "role": "user",
            "content": user_input,
        })

        reply = run_agent(input_items)

        print(f"\nAgent: {reply}")


if __name__ == "__main__":
    main()
