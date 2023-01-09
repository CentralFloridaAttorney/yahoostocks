import pandas as pd
from tabulate import tabulate

from python.yahoostocks.yahoostock import YahooStock

BASE_PATH = '../../data/md/'
TICKER = 'TSLA'


class Converter:
    @staticmethod
    def pandas_df_to_markdown_table(df):
        # Dependent upon ipython
        # shamelessly stolen from https://stackoverflow.com/questions/33181846/programmatically-convert-pandas-dataframe-to-markdown-table
        from IPython.display import Markdown, display
        fmt = ['---' for i in range(len(df.columns))]
        df_fmt = pd.DataFrame([fmt], columns=df.columns)
        df_formatted = pd.concat([df_fmt, df])
        #display(Markdown(df_formatted.to_csv(sep="|", index=False)))
        return Markdown(df_formatted.to_csv(sep="|", index=False))
    #     return df_formatted

    @staticmethod
    def df_to_markdown(df, y_index=False):
        blob = tabulate(df, headers='keys', tablefmt='pipe')
        if not y_index:
            # Remove the index with some creative splicing and iteration
            return '\n'.join(['| {}'.format(row.split('|', 2)[-1]) for row in blob.split('\n')])
        return blob

if __name__ == '__main__':
    stock_data = YahooStock(TICKER)
    x_train, y_train, x_target, y_target = stock_data.get_test_train_split(
        _data=stock_data.price_frame,
        _train_start_col=3,
        _batch_size=6,
        _train_ratio=.87,
        _target_column_start=5
    )




    converter = Converter()
    stock_markdown = converter.pandas_df_to_markdown_table(x_target)
    markdown_text = stock_markdown.data
    markdown_file = open(BASE_PATH+TICKER+'_xtarget.md', 'w+')
    markdown_file.write(markdown_text)
    markdown_file.close()
    print('done!')
