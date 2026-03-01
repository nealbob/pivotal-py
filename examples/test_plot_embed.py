import pandas as pd
import pivotal

#%%
%%pivotal

python
    import os
    _ppath = os.environ.get('PIVOTAL_PATH', '.')
    data_path = _ppath + '/examples/data/data.csv'

load df :data_path

df df_plot from df
    plot line
        x "product"
        y "price"
        title "Product Prices"
        legend False

df df_bar from df
    plot bar
        x "category"
        y "quantity"
        title "Category Quantities"

df df_scatter from df
    plot scatter
        x "price"
        y "quantity"
        c "category"
        colormap "viridis"
#%%
