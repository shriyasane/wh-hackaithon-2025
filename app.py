import gradio as gr
from PIL import Image
from ml import process_images

def display_images(image_paths):
    images = [Image.open(image_path) for image_path in image_paths]
    processed_images = process_images(images)
    return processed_images

image_demo = gr.Interface(
    fn=display_images,
    inputs=gr.Files(label="Upload Images", type="filepath"),
    outputs=gr.Gallery(label="Uploaded Images"),
    theme="glass"
)

image_demo.launch(share=True)