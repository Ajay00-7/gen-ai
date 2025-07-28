!pip install -q gradio transformers

import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "pszemraj/led-large-book-summary"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def simplify_text(text):
    input_text = "summarize: " + text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=100, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

gr.Interface(
    fn=simplify_text,
    
    inputs=gr.Textbox(lines=5, label="Complex English Sentence"),
    outputs=gr.Textbox(lines=3, label="Simplified English"),
    title="✅ English Text Simplifier",
    description="Simplifies long, complex English sentences into shorter and simpler ones. Output is always English."
).launch(share=True)