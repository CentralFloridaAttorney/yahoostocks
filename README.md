# Why use yahoostocks?

    Get a stock's historical market data: 

        tsla_stock = YahooStock('TSLA')

    Create testing and training sets for machine learning: 

        x_train, y_train, x_target, y_target = stock_object.get_test_train_split(
            _data=stock_object.price_frame,
            _train_end_col=3,
            _batch_size=6,
            _train_ratio=.87,
            _target_column_start=5
        )

_yahoostock is not affiliated with the company Yahoo(exclamation) or Tesla_

# tsla_stock.price_frame:

    date|high|low|open|close|volume|adjclose|formatted_date
    
    ---|---|---|---|---|---|---|---

    1277818200|1.6666669845581055|1.1693329811096191|1.2666670083999634|1.5926669836044312|281494500|1.5926669836044312|2010-06-29
    
    1277904600|2.0280001163482666|1.553333044052124|1.7193330526351929|1.5886670351028442|257806500|1.5886670351028442|2010-06-30
    
    1277991000|1.7280000448226929|1.3513330221176147|1.6666669845581055|1.4639999866485596|123282000|1.4639999866485596|2010-07-01
    
    1278077400|1.5399999618530273|1.24733304977417|1.5333329439163208|1.2799999713897705|77097000|1.2799999713897705|2010-07-02

_Tesla's stock market data is never guaranteed to be accurate_

### x_train:

    date|high|low

    ---|---|---
    
    1277818200|1.6666669845581055|1.1693329811096191
    
    1277904600|2.0280001163482666|1.553333044052124
    
    1277991000|1.7280000448226929|1.3513330221176147
    
    1278077400|1.5399999618530273|1.24733304977417

### x_target:

    volume

    ---
    
    281494500
    
    257806500
    
    123282000
    
    77097000


### y_train:

    date|high|low
    
    ---|---|---

    1604932200|150.8333282470703|140.3333282470703
    
    1605018600|140.02999877929688|132.00999450683594
    
    1605105000|139.56666564941406|136.86000061035156
    
    1605191400|141.0|136.5066680908203

### y_target:

    volume
    
    ---
    
    104499000
    
    90852600
    
    52073100
    
    59565300
