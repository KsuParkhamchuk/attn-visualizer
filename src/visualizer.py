"""Attention visualization using Plotly"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict


class AttentionVisualizer:

    def __init__(self):
        self.colors = px.colors.qualitative.Set3

    def create_attention_plot(
        self, attention_data: Dict, layer: int, head: int
    ) -> go.Figure:
        tokens = attention_data["tokens"]
        attention_matrix = attention_data["attention"][layer][head]

        clean_tokens = [
            token.replace("Ġ", "") if token.startswith("Ġ") else token
            for token in tokens
        ]

        fig = go.Figure()

        for i in range(len(tokens)):
            for j in range(len(tokens)):
                attention_weight = attention_matrix[i][j]

                if attention_weight > 0.01:
                    line_width = max(1, attention_weight * 5)
                    opacity = min(1.0, attention_weight * 2)

                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[j, i],
                            mode="lines",
                            line=dict(
                                width=line_width, color=f"rgba(31, 119, 180, {opacity})"
                            ),
                            hovertemplate=f"<b>{clean_tokens[j]} → {clean_tokens[i]}</b><br>Attention: {attention_weight:.3f}<br>Click to highlight all connections<extra></extra>",
                            showlegend=False,
                            legendgroup=f"token_{j}",
                            name=f"{clean_tokens[j]}_connections",
                        )
                    )

        for idx, token in enumerate(clean_tokens):

            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[len(tokens) - 1 - idx],
                    mode="text+markers",
                    text=[token],
                    textposition="middle left",
                    marker=dict(
                        size=12,
                        color="lightblue",
                        opacity=0.7,
                        line=dict(width=2, color="blue"),
                    ),
                    showlegend=False,
                    hovertemplate=f"<br>Source token<extra></extra>",
                    legendgroup=f"token_{idx}",
                    name=f"{token}_source",
                    textfont=dict(color="black"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[1],
                    y=[len(tokens) - 1 - idx],
                    mode="text+markers",
                    text=[token],
                    textposition="middle right",
                    marker=dict(
                        size=12,
                        color="lightgreen",
                        opacity=0.7,
                        line=dict(width=2, color="green"),
                    ),
                    showlegend=False,
                    hovertemplate=f"Target token<br>Shows incoming attention<extra></extra>",
                    name=f"{token}_target",
                    textfont=dict(color="black"),
                )
            )

        fig.update_layout(
            title=f"Layer {layer}, Head {head+1}",
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, range=[-0.3, 1.3]
            ),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            width=400,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="white",
        )

        return fig
