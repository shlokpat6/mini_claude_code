from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

messages = []

while True:

    user_input = input("You: ")
    if user_input.strip().lower() in ("exit", "quit"):
        break

    messages.append({"role": "user", "content": user_input})

    # Using the older chat.completions api rather tahn the responses API as the list approach for memory is easier to implement.
    response = client.chat.completions.create(
        model = "gpt-6-luna",
        messages = messages,
    )

    reply = response.choices[0].message.content
    messages.append({"role":"assistant", "content": reply})
    print("Bot", reply)