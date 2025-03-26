from PIL import Image
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def process_images(images):
    return None

def call_api(messages):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    return messages