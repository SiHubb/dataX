import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_table

from botorch.models.gp_regression import SingleTaskGP
from gpytorch.constraints import GreaterThan
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.test_functions import Hartmann

from gpytorch.mlls import ExactMarginalLogLikelihood

from torch import device as dvc
import torch

fileLoc = 'data/logs1.csv'
test_data = pd.read_csv(fileLoc)

df = px.data.iris()
fig = px.parallel_coordinates(test_data, color="mix",
                              dimensions=['p1', 'p2', 'p3', 'p4', 'mix'],
                              color_continuous_scale=px.colors.diverging.Earth,
                              color_continuous_midpoint=0.15)

external_stylesheets = [dbc.themes.BOOTSTRAP]
# external_stylesheets = [
#     {
#         "href": "https://fonts.googleapis.com/css2?"
#         "family=Poppins:wght@300&display=swap",
#         "rel": "stylesheet",
#     },
# ]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dbc.Row([
        dbc.Col('upstream logo here', width=2),
        dbc.Col('Broad and deep simulation and optimisation services', width=4),
        dbc.Col('caeClouds logo here', width=2),
        dbc.Col('The CAE podcast delving in to the breadth of toolsets', width=4),
    ]),
    dbc.Row([
        dbc.Col(children=html.Img(src=app.get_asset_url('DATAx.png')), width=4),
        dbc.Col('Upload or create an original dataset, explore relationships and '
                'get recommended next parameters to test', width=8),
    ]),
    dbc.Row([
        html.P('You are looking at the data in: ' + fileLoc)
    ]),
    dbc.Row([
        dbc.Col(
            dash_table.DataTable(
                data=test_data.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in test_data.columns]
            ), width=8
        )
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig), width=10)
    ]),
    dcc.Dropdown(
        id='p_dropdown',
        value='p1',
        options=[{'label': Parameter, 'value': Parameter} for Parameter in test_data.columns]
    ),
    dcc.Graph(id='scatter_graph'),
    dcc.Dropdown(
        id='x_bayes',
        value='p1',
        options=[{'label': Parameter, 'value': Parameter} for Parameter in test_data.columns],
        multi=True
    ),
    dcc.Dropdown(
        id='obj_bayes',
        multi=True
    ),
    dbc.Button("Confirm options", id='button'),
    html.P(id='Results')

])


@app.callback(Output('scatter_graph', 'figure'),
              Input('x_bayes', 'value'))
def plotonevar(choose_x):
    localfig = go.Figure()
    if type(choose_x) == str:
        localfig.add_scatter(x=test_data[choose_x], y=test_data['mix'])
    elif type(choose_x) == list:
        for i in choose_x:
            localfig.add_scatter(x=test_data[i], y=test_data['mix'])
    return localfig


@app.callback(Output('obj_bayes', 'options'),
              Input('x_bayes', 'value'))
def othercols(x_bayes_items):
    return [{'label': i, 'value': i} for i in test_data.columns if i not in x_bayes_items]


@app.callback(Output('Results', 'children'),
              Input('button', 'n_clicks'),
              State('x_bayes', 'value'),
              State('obj_bayes', 'value'))
def dobayes(n_clicks, x, obj):
    if not n_clicks:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate
    results = 'Cols for x: ' + str(x) + ' Cols for obj: ' + str(obj)

    # set to run on the cpu
    device = dvc("cpu")

    # configure inputs x and objs from inputs
    if type(x) == str:
        train_x = torch.tensor(test_data[x])
        print(train_x)
    if type(x) == list:
        train_x = torch.tensor(test_data[x[0]])
        for i in range(1, len(x)):
            train_x = torch.cat((train_x, torch.tensor(test_data[x[i]])), 0)
        print(train_x)

    train_obj = torch.tensor(test_data[obj[0]])
    for i in range(1, len(obj)):
        train_obj = torch.cat((train_obj, torch.tensor(test_data[x[i]])), 0)

    return train_obj


if __name__ == '__main__':
    app.run_server(debug=True)
