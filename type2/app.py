import gradio as gr
from agent.graph import app
from langchain_core.messages import HumanMessage

chat_history = []
ui_history = []

# reset button to clear all history
def reset():
    global chat_history, ui_history
    chat_history = []
    ui_history = []
    return ui_history, ''

# chat fn
def chat(user_input):
    global chat_history, ui_history

    # add user message to history
    chat_history.append(HumanMessage(content=user_input))
    
    # call langgraph app
    result = app.invoke({'messages': chat_history})
    chat_history = result['messages']

    # get latest ai message
    ai_output = chat_history[-1].content

    # append ai message to UI history for display
    ui_history.append((user_input, ai_output))

    return ui_history, ''

# gradio chatbot interface
with gr.Blocks() as demo:
    gr.Markdown('### Smart Travel Insurance Agent')
    gr.Markdown('Start the chat with your travel profile and let AI-powered agent get you the best plan!')

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder='Tell me about your trip and needs...')
    msg.submit(chat, inputs=msg, outputs=[chatbot,msg])
    
    clear_btn = gr.Button('Clear Chat')
    clear_btn.click(reset, outputs=[chatbot,msg])
    
if __name__ == '__main__':
    demo.launch(share=False, auth=('user','pwd'))