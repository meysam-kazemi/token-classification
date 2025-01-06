# app.py

from src.classifier import Classifer
import gradio as gr

classifier = Classifer()

def chat(text, history):
    history = history or []
    history.append((text, classifier(text)))
    return history, history


iface = gr.Interface(
    chat,
    ["text", 'state'],
    ["chatbot", 'state'],
    # allow_screenshot=False,
    allow_flagging="never",
)
iface.launch()
