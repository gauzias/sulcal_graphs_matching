import plotly.graph_objs as go


def error_plot(x, y, y_lower, y_upper, line_label='Measurement', color='rgb(180, 180, 180)'):
    fig_content = [
        go.Scatter(
            name=line_label,
            x=x,
            y=y,
            mode='lines',
            line=dict(color=color, width=4),
        ),
        go.Scatter(
            name='Upper Bound',
            x=x,
            y=y_upper,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=x,
            y=y_lower,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ]
    return fig_content
