import gradio as gr
from PIL import Image
from ml import process_images, call_api

def display_images(image_paths):
    images = [Image.open(image_path) for image_path in image_paths]
    predictions = process_images(images)
    
    prediction_data = [
        {"Image": f"Image {i+1}", "Class": pred["class"], "Confidence": f"{pred['confidence']:.2f}"}
        for i, pred in enumerate(predictions)
    ]

    return images, prediction_data

def chatbot(messages, user_message):
    messages.append({"role": "user", "content": user_message})
    messages = call_api(messages)
    return messages

image_demo = gr.Interface(
    fn=display_images,
    inputs=gr.Files(label="Upload Images", type="filepath"),
    outputs=[
        gr.Gallery(label="Uploaded Images"),
        gr.DataFrame(headers=["Image", "Class", "Confidence"], label="Predictions"), 
    ], 
    theme="glass"
)

chatbot_demo = gr.Blocks(theme="glass")

with chatbot_demo:
    chat_history = gr.Chatbot(type="messages")
    with gr.Column():
        user_input = gr.Textbox(lines=2, placeholder="Enter your message here...")
        submit_button = gr.Button("Send")
    
    def submit_message(messages, user_message):
        return chatbot(messages, user_message)
    
    submit_button.click(submit_message, [chat_history, user_input], chat_history)

demo = gr.TabbedInterface([image_demo, chatbot_demo], ["Image Upload", "Chatbot"], theme="glass")
demo.launch()