import gradio as gr

with gr.Blocks() as yahoo_app:
    with gr.Tab("Main"):
        with gr.Row():
            submit_btn = gr.Button(value="Submit")



yahoo_app.launch()