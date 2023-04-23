import gradio as gr
import pandas

from python.yahoostocks.yahoostock import YahooStock

CASE_GRID_COLUMNS = ["CASE NUMBER", "PARTIES", "COURT / DIV", "CASE TYPE", "JUDGE", "STATUS", "FILE DATE"]


def get_default_dataframe():
    d = {'default': {CASE_GRID_COLUMNS[0]: "00-DR-0000",
                     CASE_GRID_COLUMNS[1]: "John Doe -vs- Jane Doe",
                     CASE_GRID_COLUMNS[2]: "DR",
                     CASE_GRID_COLUMNS[3]: "DR",
                     CASE_GRID_COLUMNS[4]: "None",
                     CASE_GRID_COLUMNS[5]: "Closed",
                     CASE_GRID_COLUMNS[6]: "02-02-2022"}}
    data_frame = pandas.DataFrame(d, dtype='str').fillna("default")
    data_frame = data_frame.transpose()
    return data_frame

def get_ticker_history(_ticker):
    this_stock = YahooStock(_ticker)
    ticker_history = this_stock.price_frame
    return ticker_history


with gr.Blocks() as stock_app:
    with gr.Tab("Available Stocks"):
        with gr.Row():
            btn_update_stock_list = gr.Button(value="Update Stock List")
            btn_set_default_stock_list = gr.Button(value="Reset Stock List")
        with gr.Row():
            dataframe_stock_list = gr.Dataframe(get_default_dataframe())
    with gr.Tab("Stock History"):
        with gr.Row():
            btn_update_stock_history = gr.Button(value="Update Stock List")
            textbox_ticker = gr.Textbox(value="SCHW")
        with gr.Row():
            dataframe_stock_history = gr.Dataframe(get_default_dataframe())
    btn_update_stock_history.click(fn=get_ticker_history, inputs=[textbox_ticker], outputs=[dataframe_stock_history])
stock_app.queue()
stock_app.launch()