from PIL import Image
from dotenv import load_dotenv
import os
from openai import OpenAI
from predict import predict

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

def process_images(images):
    predictions = []
    for image in images: 
        predicted_class, confidence = predict(image)
        predictions.append({"class": predicted_class, "confidence": confidence})
    return predictions

def call_api(messages):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    response = completion.choices[0].message.content
    messages.append({"role": "assistant", "content": response})
    return messages