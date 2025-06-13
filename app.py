from utils import extract_text_from_image
from geometryLLM import solver
import gradio as gr

def process_image_and_answer(image):
    try:
        text = extract_text_from_image(image)
        if not text:
            return "Could not read image"
            
        answer = solver.solve(text)
        return f"Question: {text}\nAnswer: {answer}"
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=process_image_and_answer,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Geometry Solver",
    description="Upload an image of a geometry problem"
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)