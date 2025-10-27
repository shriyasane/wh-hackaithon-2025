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
    response = {"role": "assistant", "content": ""}
    messages.append(response)

    for chunk in call_api(messages):
        response["content"] += chunk
        yield messages.copy()

with gr.Blocks(css="styles.css") as demo:
    gr.Markdown("# Dyscover")
    with gr.Tab("Image Upload"):
        gr.Markdown("## A tool to help teachers easily flag students that may have dyslexia through their writing samples.")
        gr.Markdown("### Upload images of your students' writing samples and we'll predict if they show signs of dyslexia. Feel free to head to the Chatbot tab to ask any questions about dyslexia and what tools you can use to help improve your students writing.")

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
        gr.Markdown("Feel free to ask any questions about dyslexia and what tools you can use to help improve your students writing.")
        chat_history = gr.Chatbot(type="messages", height=450)
        with gr.Column():
            user_input = gr.Textbox(lines=2, placeholder="Enter your message here...")
            submit_button = gr.Button("Send")

            with gr.Row():
                prompt_1 = gr.Button("What are signs of dyslexia in k-2?")
                prompt_2 = gr.Button("How can I help a student with dyslexia?")
                prompt_3 = gr.Button("What are some lesson plans for dyslexic students?")

        submit_button.click(chatbot, [chat_history, user_input], chat_history)
        prompt_1.click(chatbot, [chat_history, gr.State("What are signs of dyslexia in k-2?")], chat_history)
        prompt_2.click(chatbot, [chat_history, gr.State("How can I help a student with dyslexia?")], chat_history)
        prompt_3.click(chatbot, [chat_history, gr.State("What are some lesson plans for dyslexic students?")], chat_history)

demo.launch()