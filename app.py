import gradio as gr
from PIL import Image
from ml import process_images, call_api

def display_images(image_paths):
    images = [Image.open(image_path) for image_path in image_paths]  
    predictions = process_images(images)  

    dys = [
        (image, f"Class: {pred['class']}, Confidence: {pred['confidence']:.2f}")
        for image, pred in zip(images, predictions)
        if pred['class'] == 'dyslexic'
    ]
    control = [
        (image, f"Class: {pred['class']}, Confidence: {pred['confidence']:.2f}")
        for image, pred in zip(images, predictions)
        if pred['class'] == 'non-dyslexic'
    ]

    return dys, control  

def chatbot(messages, user_message):
    messages.append({"role": "user", "content": user_message})
    messages = call_api(messages)
    return messages

with gr.Blocks(theme="glass") as demo:
    with gr.Tab("Image Upload"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Dyslexic Predictions")
                dys_gallery = gr.Gallery(label="Dyslexic Images with Predictions")

            with gr.Column():
                gr.Markdown("### Non-Dyslexic Predictions")
                control_gallery = gr.Gallery(label="Non-Dyslexic Images with Predictions")

        upload_images = gr.Files(label="Upload Images", type="filepath")

        upload_images.change(
            display_images,
            inputs=[upload_images],
            outputs=[dys_gallery, control_gallery],
        )

    with gr.Tab("Chatbot"):
        chat_history = gr.Chatbot(type="messages", height=500)
        with gr.Column():
            user_input = gr.Textbox(lines=2, placeholder="Enter your message here...")
            submit_button = gr.Button("Send")

            with gr.Row():
                prompt_1 = gr.Button("What are signs of dyslexia in k-2?")
                prompt_2 = gr.Button("How can I help a student with dyslexia?")
                prompt_3 = gr.Button("What are some lesson plans for dyslexic students?")

        def submit_message(messages, user_message):
            return chatbot(messages, user_message)

        submit_button.click(submit_message, [chat_history, user_input], chat_history)
        prompt_1.click(submit_message, [chat_history, gr.State("What are signs of dyslexia in k-2?")], chat_history)
        prompt_2.click(submit_message, [chat_history, gr.State("How can I help a student with dyslexia?")], chat_history)
        prompt_3.click(submit_message, [chat_history, gr.State("What are some lesson plans for dyslexic students?")], chat_history)

demo.launch()