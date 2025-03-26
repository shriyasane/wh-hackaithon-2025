from PIL import Image
from dotenv import load_dotenv
import os 
from openai import OpenAI

client = OpenAI()

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

def process_images(images): 
    return None 

def call_api(): 
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "Write a one-sentence bedtime story about a unicorn."
    }]  
    )

    print(completion.choices[0].message.content)