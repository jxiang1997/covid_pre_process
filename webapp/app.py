import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

# Constants
img_width = 1664
img_height = 2048
scale_factor = 0.25

# App backend
app = dash.Dash()


# App frontend
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=[0, img_width * scale_factor],
        y=[0, img_height * scale_factor],
        mode="markers",
        marker_opacity=0
    )
)

# Configure axes
fig.update_xaxes(
    visible=False,
    range=[0, img_width * scale_factor]
)

fig.update_yaxes(
    visible=False,
    range=[0, img_height * scale_factor],
    # the scaleanchor attribute ensures that the aspect ratio stays constant
    scaleanchor="x"
)

# Add image
fig.add_layout_image(
    dict(
        x=0,
        sizex=img_width * scale_factor,
        y=img_height * scale_factor,
        sizey=img_height * scale_factor,
        xref="x",
        yref="y",
        opacity=1.0,
        layer="below",
        sizing="stretch",
        source=app.get_asset_url('fixed.png'))
)

fig.update_layout(clickmode='event+select')

# Configure other layout
fig.update_layout(
    width=img_width * scale_factor,
    height=img_height * scale_factor,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
)

app.layout = html.Div(children=[
    html.H1(
        children='Mammogram Alignement Annotation Tool',
        style={
            'textAlign': 'center',
            'color': '#111111'
        }
    ),
    html.Div([
        dcc.Graph(
            id='fixed',
            figure=fig,
            config={
                'displayModeBar': False
            }
    )], style={'display': 'inline-block', 'margin':'10px'}),  
    html.Div([
        dcc.Graph(
            id='moving',
            figure=fig,
            config={
                'displayModeBar': False
            }
    )], style={'display': 'inline-block', 'margin': '10px'})    
], style={'textAlign': 'center'})


@app.callback(dash.dependencies.Output('moving', 'figure'),
              [dash.dependencies.Input('moving', 'hoverData')])
def capture(hoverData):
    print('click', hoverData)
    if hoverData:
        print(hoverData)


if __name__ == '__main__':
    app.run_server()
