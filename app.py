import dash
import dash_html_components as html
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
                dbc.Row([
                    dbc.Col('upstream logo here', width=2),
                    dbc.Col('Borad and deeep simulation and optimisation services', width=4),
                    dbc.Col('caeClouds logo here', width=2),
                    dbc.Col('The CAE podcast delving in to the breadth of toolsets', width=4),
                ]),
                dbc.Row([
                    dbc.Col(children=html.Img(src=app.get_asset_url('DATAx.png')), width=4),
                    dbc.Col('Upload or create an original dataset, explore relationships and '
                    'get recommended next parameters to test', width=8),
                ]),
                dbc.Tabs([
                    dbc.Tab([

                    ],label='Upload/Generate Data'),
                    dbc.Tab([

                    ],label='Explore Data'),
                    dbc.Tab([

                    ],label='Recommend Data Exploration')
                ])

            ])

if __name__ == '__main__':
    app.run_server(debug=True)