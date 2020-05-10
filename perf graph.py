import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

x = ["v1", "v2", "v2.5", "v3"]
y_avg_success = np.array([[58.9, 10.9], [79.0, 38.5], [78.7, 38.9], [83.9, 49.3]])
y_avg_run_time = np.array([[3.5, 10.6], [7.3, 26.2], [8.4, 31.8], [26.1, 276.2]])


for i in range(2):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Average success rate", "Average running time")
    )
    fig.add_trace(
        go.Scatter(x=x, y=y_avg_success[:, i]),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=y_avg_run_time[:, i]),
        row=1, col=2
    )
    fig.update_xaxes(title_text="Versions of algorithm", row=1, col=1)
    fig.update_yaxes(title_text="Total average success rate (in %)", range=[0, 100], row=1, col=1)
    fig.update_xaxes(title_text="Versions of algorithm", row=1, col=2)
    fig.update_yaxes(title_text="Total average running time (in ms)", row=1, col=2)

    if i == 0:
        fig.update_layout(height=600, width=800, title_text='Average success with a 16x16 grid, and 40 mines')
    else:
        fig.update_layout(height=600, width=800, title_text='Average success with a 16x32 grid, and 99 mines')
    fig.show()
