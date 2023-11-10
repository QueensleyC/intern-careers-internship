from dash import Dash, dcc, html, Output, Input, dash_table, clientside_callback
from dash.dependencies import Input, Output, State
from dash_iconify import DashIconify
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

import datetime
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# Read in data
df = pd.read_csv('youtubers_df_cleaned.csv')
df = df.reset_index()

# Combine important columns
def combine_features(data):
    features = []
    for i in range(0, data.shape[0]):
        features.append(data["Categories"][i] + " " + data["Country"][i])
    return features
    
df["combined_features"] = combine_features(df)

# Instantiate app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    children = [
        html.H5('Select YouTuber'),
        dcc.Dropdown(options = df.Username.values, value = df.Username.values[0], id='ytuber-username', clearable=False),
        html.Br(), 

        html.H5('These are the top 5 recommended channels:'),
        html.Div(id = 'r1'),
        html.Div(id = 'r2'),
        html.Div(id = 'r3'),
        html.Div(id = 'r4'),
        html.Div(id = 'r5'),

    ]
)

@app.callback(
    Output('r1', 'children'),
    Output('r2', 'children'),
    Output('r3', 'children'),
    Output('r4', 'children'),
    Output('r5', 'children'),
    Input('ytuber-username','value')
)
def recommend(account):
    recommendations = []

    cm = CountVectorizer().fit_transform(df["combined_features"])
    cs = cosine_similarity(cm)

    # Find the index of the YouTuber the user likes
    ytuber_id = df[df.Username == account]["index"].values[0]

    # Create a list of tuples in the form (username, similarity score)
    scores = list(enumerate(cs[ytuber_id]))

    # Sort the list of similar books in descending order
    sorted_scores = sorted(scores, key = lambda x:x[1], reverse = True)
    sorted_scores = sorted_scores[1:]

    for i in sorted_scores:
        if i[0] == ytuber_id:
            sorted_scores.remove(i)
    
    # Create a loop to store the first 5 most similar accounts    
    j = 0
    for item in sorted_scores:
        accounts = df[df['index'] == item[0]]["Username"].values[0]
        recommendations.append(accounts)
        j = j+1
        if j >= 5:
            break
    r1 = recommendations[0]
    r2 = recommendations[1]
    r3 = recommendations[2]
    r4 = recommendations[3]
    r5 = recommendations[4]

    return r1,r2,r3,r4,r5


if __name__ == '__main__':
    app.run_server(debug = True)
