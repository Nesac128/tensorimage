import dash
import dash_core_components as dcc
import dash_html_components as html


nav_bar = html.Nav(className="op nav-bar", children=[
        html.A('Training', className="op nav-bar train-op", href='/train/',
               style={
                   'text-decoration': 'None',
                   'color': 'yellow',
                   'padding-left': '10px',
                   'margin-top': '0px'
               }),
        html.A('Classication', className="op nav-bar classify-op", href='/classify/',
               style={
                   'text-decoration': 'None',
                   'color': 'yellow',
                   'padding-left': '10px',
                   'margin-top': '0px'
               }),
        ], style={
            'background-color': 'lightblue',
            'margin-bottom': '30px',
        })


training_app = dash.Dash(url_base_pathname='/train/')
training_app.layout = html.Div(children=[
    nav_bar,
    dcc.Graph(
        id='accuracy-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [0.2, 0.7, 0.95], 'type': 'line', 'name': 'Training accuracy'},
                {'x': [1, 2, 3], 'y': [0.15, 0.5, 0.8], 'type': 'line', 'name': 'Testing accuracy'},
            ],
            'layout': {
                'title': 'Training accuracy'
            }
        },
        style={'height': '600px',
               'width': '1200px'}
    ),
])


if __name__ == '__main__':
    training_app.run_server()
