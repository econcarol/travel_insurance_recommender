import gradio as gr
from rag_engine import recommender

def handle_submit(user_input):
    return recommender(user_input)

demo = gr.Interface(
    fn = handle_submit,
    inputs = gr.Textbox(
        lines = 5, 
        placeholder="e.g. I'm 63, traveling to Thailand for 3 weeks. I have hypertension."
    ),
    outputs = "text",
    title = "Smart Travel Insurance Recommender",
    description = "Enter your travel profile. Get an AI-powered insurance suggestion based on your needs."
)

if __name__ == "__main__":
    demo.launch(share=False, auth=("user", "pwd"))