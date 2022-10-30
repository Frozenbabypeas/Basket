import streamlit as st
import pandas as pd

#set up layout
page_title = 'Basket Analysis'
page_icon = ':honey_pot:'
layout = 'centered'

st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.header('Welcome to this test example!!')
    st.text('Identifing behaviour associations to loon into common items in the basket')

with dataset:
    st.header('Super Store Dataet')
    st.text('Kaggle Data set that can be found at Kaggle')

    df = pd.read_excel('Global Superstore.xls')
    df = df.sort_values('Order Date').reset_index()
    dfsample = df[['Order Date', 'Category', 'Sub-Category', 'Product Name', 'Profit']]
    st.write(dfsample)

    st.subheader('Sales trend through dataset')
    st.line_chart(df, x = 'Order Date', y = 'Sales')

with features:
    st.header('The feautures I created')

    st.markdown('* **First Feature:**  I created this Feature beacuse ...')
    st.markdown('* **First Feature:**  I created this Feature beacuse ...')
    st.markdown('* **First Feature:**  I created this Feature beacuse ...')
    

with model_training:
    st.header('Time to train the model!')
    st.text('Choose the Parameters that work for you')

    sel_col, disp_col = st.columns(2) # creating two columns within the container, one with a selection, the other with the graph

    max_depth = sel_col.slider('what should be the max_depth of the model?', min_value = 10, max_value = 100, value = 50, step = 10)

    n_estimators = sel_col.selectbox('How many tress should there be?', options = [100,200,300, 'No Limit'])

    input_feature = sel_col.text_input('Which feature should ybeused as the input feature?', 'testing value')