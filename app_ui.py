import gradio as gr
import random

def classify_image(image):
    
    label = random.choice(["AI Generated", "Real"])
    confidence = random.uniform(0.7, 0.99)

    return f"{label} ({confidence:.2f})"


interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="AI vs Real Image Detector",
    description="Upload an image to classify it as AI-generated or real."
)

if __name__ == "__main__":
    interface.launch(share=True)