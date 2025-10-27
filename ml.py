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
    if len(messages) == 1:
        messages.append({
            "role": "system", 
            "content": "You are a tutor/lesson plan creator to help teachers of children k-2nd grade that have been flagged as potentially having dyslexia, after having their handwritings analyzed. Answer the questions from the teacher with personalized lesson plans to help the students master the writing skills they're struggling with."
        })
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=messages, 
        stream=True
    )
    response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content: 
            response += content
            yield content
    
    messages.append({"role": "assistant", "content": response})