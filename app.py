import gradio as gr
from PIL import Image
from ml import process_images, call_api

def display_images(image_paths):
    images = [Image.open(image_path) for image_path in image_paths]
    processed_images = process_images(images)
    return processed_images

def chatbot(messages, user_message):
    messages.append({"role": "user", "content": user_message})
    messages = call_api(messages)
    return messages

image_demo = gr.Interface(
    fn=display_images,
    inputs=gr.Files(label="Upload Images", type="filepath"),
    outputs=gr.Gallery(label="Uploaded Images"),
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