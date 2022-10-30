# set up streamlit
import streamlit as st
from streamlit_option_menu import option_menu

#set up layout
page_title = 'Basket Analysis'
page_icon = ':honey_pot:'
layout = 'centered'

st.set_page_config(page_title = page_title, page_icon = page_icon, layout = layout)

# get data from data.py
from sre_parse import CATEGORIES
from urllib import request
from data import df

# set up environment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlxtend
from itertools import permutations
import seaborn as sns
import datetime

# graphing data
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

# basket analysis
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori

# remove warnings
import warnings
warnings.filterwarnings('ignore')

# network graph
import networkx as nx

st.title(page_title)

Summary_content = st.container()
Summary_detail = st.container()

with Summary_content:
        st.header('Store Details')
        st.write("Pick the below drill down items to isolate the data")
        sum_col, sum_year = st.columns(2) # creating two columns within the container

        countries = sum_col.selectbox('Select Country', options = df['Country'].drop_duplicates().sort_values())
        years = sum_year.selectbox('Select Year', options = df['Order year'].drop_duplicates().sort_values())

        input_country_selection = df.loc[df['Country'] == countries][df['Order year'] == years]
        freq_items = input_country_selection['Sub-Category'].value_counts() 

        # plot most sold items
        fig = px.bar(freq_items, title = 'Top Sold Items', color = freq_items, labels = {
            'index' : 'Item',
            'values' : 'Quantity',
            'lift' : 'Lift'
        })
        figure1 = fig.update_layout(title_x = 0.5, title_y = 0.86)
        st.write(figure1)
        #figure 1
        
        st.write(f"In {countries} the most items bought is in the category of {freq_items.index[0]}, with {freq_items[0]} sold in the year of {years}")

st.header('Consumer Behaviour')
st.subheader('Product Association Rule')
st.write('Using an association algorithm, we can combine the different sets of item purchases and thier frequency in a basket.  Association Rules are widely used to analyse transaction based data to identify sets and associations.  We can drill into the data based on country and period.  Select the Category to see the associated categories bought.')

cat_select = st.selectbox('Select Category', options = input_country_selection['Sub-Category'].drop_duplicates().sort_values())

#--------------------list items------------------------
user_id = df['Customer ID'].unique()
items = [list(df.loc[df['Customer ID'] == id, 'Sub-Category']) for id in user_id]

#--------------------create a item matrix------------------------
TE = TransactionEncoder()
TE.fit(items)
item_transformed = TE.transform(items)
item_matrix = pd.DataFrame(item_transformed, columns = TE.columns_)

#--------------------get the support value by Apriori algorithm------------------------
freq_items = apriori(item_matrix, min_support=0.01, use_colnames=True, max_len=2) # we can increase the basket size by changing the max_len

#--------------------create a dataframe with product support, confidence , and lift values------------------------
rules = association_rules(freq_items, metric = "confidence", min_threshold = 0)
figure2 = rules
#figure 2

#--------------------add a column for a Zhang's core------------------------
def zhangs_rule(rules):
    rule_support = rules['support'].copy()
    rule_ante = rules['antecedent support'].copy()
    rule_conseq = rules['consequent support'].copy()
    num = rule_support - (rule_ante * rule_conseq)
    denom = np.max((rule_support * (1 - rule_ante).values, 
                        rule_ante * (rule_conseq - rule_support).values), axis = 0)
    return num / denom

rules_zhangs_list = zhangs_rule(rules)
rules = rules.assign(zhang = rules_zhangs_list)

#-------------------- select items that have high support, choose it as the item for the basket analysis------------------------
rules_sel = rules[rules["antecedents"].apply(lambda x: cat_select in x)] # feature selection 3, changing chairs to different product to see the relationship
rules_sel = rules_sel.sort_values('confidence', ascending=False)

#-------------------- get the most important items that customers would buy after purchasing feature selection 3------------------------
rules_support = rules_sel['support'] >= rules_sel['support'].quantile(q = 0.95)
rules_confi = rules_sel['confidence'] >= rules_sel['confidence'].quantile(q = 0.95)
rules_lift = rules_sel['lift'] > 1
rules_zhang = rules_sel['zhang'] > 0
rules_best = rules_sel[rules_support & rules_confi & rules_lift & rules_zhang]

#-------------------- prepare the top 10 persentile items for visualization------------------------
rules_eda = rules_sel.copy(deep=True)
rules_support_eda = rules_eda['support'] >= rules_eda['support'].quantile(q = 0.9)
rules_confi_eda = rules_eda['confidence'] >= rules_eda['confidence'].quantile(q = 0.9)
rules_lift_eda = rules_eda['lift'] > 1
rules_zhang_eda = rules_eda['zhang'] > 0
rules_best_eda = rules_eda[rules_support_eda & rules_confi_eda & rules_lift_eda & rules_zhang_eda]

#-------------------- # remove the parentheses in the antecedents and consequents columns.  Needs only to be applied once------------------------
rules_best_eda['antecedents'] = rules_best_eda['antecedents'].apply(lambda a: ', '.join(list(a)))
rules_best_eda['consequents'] = rules_best_eda['consequents'].apply(lambda a: ', '.join(list(a)))

#-------------------- plot a heatmap to know how strong the association is regarding lift values------------------------
pivot_confidence = rules_best_eda.pivot(index='antecedents', columns='consequents', values='confidence')

fig = ff.create_annotated_heatmap(pivot_confidence.to_numpy().round(2), 
                                x=list(pivot_confidence.columns), 
                                y=list(pivot_confidence.index))
fig.update_layout(
    template='simple_white',
    xaxis_title='Consequents',
    yaxis_title='Antecedents',
    legend_title="Legend Title"
)
fig.update_layout(title_x=0.22, title_y=0.98)
fig.update_traces(showscale=True)
figure4 = fig.update_layout(title_x=0.22, title_y=0.98).update_traces(showscale=True)
#figure 4

#-------------------- plot the network to see the connections between the top percentile items------------------------
network_A = list(rules_best_eda["antecedents"].unique())
network_B = list(rules_best_eda["consequents"].unique())
node_list = list(set(network_A+network_B))
G = nx.Graph()
for i in node_list:
    G.add_node(i)
for i,j in rules_best_eda.iterrows():
    G.add_edges_from([(j["antecedents"],j["consequents"])])
pos = nx.spring_layout(G, k=0.5, dim=2, iterations=400)
for n, p in pos.items():
    G.nodes[n]['pos'] = p

edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')
for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        reversescale=True,
        color=[],
        size=15,
        colorbar=dict(
            thickness=10,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        ),
        line=dict(width=0)))
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color']+=tuple([len(adjacencies[1])])
    node_info = str(adjacencies[0]) +' has {} connections'.format(str(len(adjacencies[1])))
    node_trace['text']+=tuple([node_info])

fig = go.Figure(data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Connection of Selected Category',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

figure5 = fig.update_layout(title_x=0.5, title_y=0.96)
#figure 5

# nav menu
selected = option_menu(
        menu_title = None,
        options = ['Visual', 'Raw Data'],
        icons = ['bar-chart-fill', 'graph-up'],
        orientation = 'horizontal')

# Visuals
if selected == 'Visual':
    purchase_lift = st.container()
    network_lift = st.container()
    with purchase_lift:
        st.subheader('Lift Matrix')
        st.write('The lift matrix will can show us the antecednet (the category first purchased) against the consequent (the category purchases after the first purchase) with its lift, which is the ratio of consequent sale after the antecedent sale.  The higher our lift value will mean the higher the chance that the antecedent purchase will result with the consequent purchase.')
        st.write(figure4)

    with network_lift:
        st.subheader('Associated purchases')
        st.write(f'This network graph will show us the other category purchases from the {cat_select} category')
        st.write(figure5)

# Raw data menu
if selected == 'Raw Data':
    raw_data = st.container()

    with raw_data:
        st.subheader('Apriori Assocation Data')
        st.write('Raw Data on selected category')
        st.write(df)
        st.write('Top 10 percentile of categories purchased')
        st.write(rules_best_eda)