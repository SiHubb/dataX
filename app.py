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
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement

from gpytorch.mlls import ExactMarginalLogLikelihood

from torch import device as dvc
import torch

fileLoc = 'data/logs1.csv'
test_data = pd.read_csv(fileLoc)


parfig = px.parallel_coordinates(test_data, color="mix",
                              dimensions=['p1', 'p2', 'p3', 'p4', 'mix'],
                              color_continuous_scale=px.colors.diverging.Earth,
                              color_continuous_midpoint=0.15)

external_stylesheets = [dbc.themes.LUMEN]
# external_stylesheets = [
#     {
#         "href": "https://fonts.googleapis.com/css2?"
#         "family=Poppins:wght@300&display=swap",
#         "rel": "stylesheet",
#     },
# ]

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dbc.Row([
        dbc.Col(),
        dbc.Col(children=html.Img(src=app.get_asset_url('DATAx.png')),style={'textAlign': 'center'}),
        dbc.Col( ),
            ],
            className="header"),

    dbc.Row([
        dbc.Col(),
        dbc.Col(dcc.Graph(figure=parfig)),
        dbc.Col()
    ], className="parcoordsgraph"),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.P('Variable'),
                    dcc.Dropdown(
                        id='hist_drop',
                        value='p1',
                        options=[{'label': Parameter, 'value': Parameter} for Parameter in test_data.columns]
                    ),

                ], width=4),

                dbc.Col([
                    html.P('# bins'),
                    dcc.Slider(
                        id='hist_slide',
                        min=5,
                        max=15,
                        step=5,
                        included=False
                    ),

                ], width=8),
            ]),

            dbc.Row(
                dbc.Col(dcc.Graph(id='hist'))
            )
        ],width=4),

        dbc.Col([
            dbc.Row([


            ]),

            dbc.Row(
                dbc.Col(dcc.Graph(id='corr'))
            )
        ],width=4),

        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.P('x variable'),
                    dcc.Dropdown(
                        id='scat_drop',
                        value='p1',
                        options=[{'label': Parameter, 'value': Parameter} for Parameter in test_data.columns]
                        ),

                ],width=4),
                dbc.Col([
                    html.P('y variable'),
                    dcc.Dropdown(
                        id='scat_drop2',
                        value='p2',
                        options=[{'label': Parameter, 'value': Parameter} for Parameter in test_data.columns]
                        ),

                ],width=4),
                dbc.Col([
                    html.P('Colour variable'),
                    dcc.Dropdown(
                        id='scat_drop3',
                        value='mix',
                        options=[{'label': Parameter, 'value': Parameter} for Parameter in test_data.columns]
                        ),

                ],width=4),



            ]),

            dbc.Row(
                dbc.Col(dcc.Graph(id='scat'))
            )
        ],width=4),

    ]),

    dcc.Dropdown(
        id='p_dropdown',
        value='p1',
        options=[{'label': Parameter, 'value': Parameter} for Parameter in test_data.columns]
    ),

    dcc.Dropdown(
        id='x_bayes',
        #value='p1',
        options=[{'label': Parameter, 'value': Parameter} for Parameter in test_data.columns],
        multi=True
    ),
    dcc.Dropdown(
        id='obj_bayes',
        multi=True
    ),
    dbc.Button("Confirm options", id='button'),
    html.P(id='Results'),

    dash_table.DataTable(
        id='datatable',

    )
])


@app.callback(Output('scat','figure'),
              Input('scat_drop','value'),
              Input('scat_drop2','value'),
              Input('scat_drop3','value'))
def makescat(a,b,c):
    fig = px.scatter(test_data, x=a, y=b, color=c, template="simple_white")
    return fig;

@app.callback(Output('hist','figure'),
              Input('hist_drop','value'),
              Input('hist_slide','value'),
             )
def makehist(a,b):
    fig = px.histogram(test_data, x=a, nbins=b)
    return fig;

@app.callback(Output('datatable', 'columns'),
              Input('x_bayes', 'value'))
def passiton(x_bayes_items):
    if type(x_bayes_items) == list:
        return [{"name": i, "id": i} for i in x_bayes_items]

@app.callback(Output('obj_bayes', 'options'),
              Input('x_bayes', 'value'))
def othercols(x_bayes_items):
    print(type(x_bayes_items))
    if type(x_bayes_items) == list:
        return [{'label': i, 'value': i} for i in test_data.columns if i not in x_bayes_items]
    else:
        return [{'label': i, 'value': i} for i in test_data.columns]



@app.callback(Output('datatable', 'data'),
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

    if type(x) == list:
        tensSeq = []
        for i in range(0, len(x)):
            tenstr = x[i]
            locals()[tenstr] = torch.tensor(test_data[x[i]])
            tensSeq.append(locals()[tenstr])
        train_x = torch.dstack(tuple(tensSeq))[0]

    tensSeq = []
    for i in range(0, len(obj)):
        tenstr = obj[i]
        locals()[tenstr] = torch.tensor(test_data[obj[i]])
        tensSeq.append(locals()[tenstr])
    train_obj = torch.dstack(tuple(tensSeq))[0]

    #train_obj = train_obj.unsqueeze(-1)

    # print(obj)
    # print(train_x)
    # print(train_obj)

    #calculate bounds from x data limits
    l = train_x.shape[1]
    mins = [train_x[:,i].min() for i in range(l)]
    maxs = [train_x[:,i].max() for i in range(l)]
    bounds = torch.tensor([[mins],[maxs]])

    #setup model
    model = SingleTaskGP(train_x, train_obj)
    #model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(noise))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    #fit model to x data
    fit_gpytorch_model(mll)

    #select acqstn function based on number of objectives
    if train_obj.shape[1] == 1:
        #use analytic expected improvement
        EI = ExpectedImprovement(model=model, best_f=train_obj.max())

        # optimize acquisition function
        params, _ = optimize_acqf(
            acq_function=EI,
            bounds=bounds,
            num_restarts=20,
            raw_samples=100,
            q=1,
            options={},
        )

    elif train_obj.shape[1] > 1:
        #use hypervolume improvement
        #calculate ref point as minimum of all objectives
        k = train_obj.shape[1]
        ref_point=torch.tensor([train_obj[:,i].min() for i in range(k)])
        #print(ref_point)
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_obj)

        sampler = SobolQMCNormalSampler(num_samples=512)

        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),  # use known reference point
            partitioning=partitioning,
            sampler=sampler,
        )
        # optimize
        params, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=512,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
            sequential=True,
        )

    temp = [{x[i]: float(params[0][i]) for i in range(len(x))}]

    return temp



if __name__ == '__main__':
    app.run_server(debug=True)
