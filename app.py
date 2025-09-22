"""Streamlit web app for attention visualization"""

import streamlit as st
from src.attn_extractor import AttentionExtractor
from src.visualizer import AttentionVisualizer

st.set_page_config(
    page_title="GPT-2 Attention Visualizer", page_icon="🔍", layout="wide"
)


def load_model(model_name: str = "gpt2"):
    """Load model once and cache it"""
    return AttentionExtractor(model_name)


def extract_attention_cached(text: str, model_name: str = "gpt2"):
    """Cache attention extraction for same text"""
    extractor = load_model(model_name)
    return extractor.extract_attention(text)


def main():
    sec1, sec2, sec3 = st.columns([1, 8, 1])
    with sec2:
        st.title("🔍 Attention Visualizer")
        st.markdown("*Visualize attention patterns in GPT-2 like BertViz*")

        st.header("Controls")

        default_text = "The quick brown fox jumps over the lazy dog"
        text = st.text_area(
            "Input Text:",
            value=default_text,
            height=100,
            help="Enter text to analyze attention patterns",
        )
        model = st.selectbox("Model", ["gpt2", "bert-base-uncased"])
        if text.strip():
            with st.spinner("Loading model and extracting attention..."):
                attention_data = extract_attention_cached(text, model)

            col1, col2 = st.columns(2)

            with col1:
                layer = st.selectbox(
                    "Layer:",
                    range(attention_data["num_layers"]),
                    index=0,
                    help="Select transformer layer (0-11)",
                )

            with col2:
                head = st.selectbox(
                    "Head:",
                    range(attention_data["num_heads"]),
                    index=0,
                    help="Select attention head (0-11)",
                )

            view_mode = st.radio(
                "View Mode:",
                ["Single View", "All Heads"],
                help="Choose visualization layout",
            )

            visualizer = AttentionVisualizer()

            if view_mode == "Single View":
                st.subheader(f"Layer {layer}, Head {head}")
                fig = visualizer.create_attention_plot(attention_data, layer, head)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.subheader(f"All Heads - Layer {layer}")

                for row in range(3):
                    cols = st.columns(4)
                    for col_idx in range(4):
                        head_idx = row * 4 + col_idx

                        if head_idx < attention_data["num_heads"]:
                            with cols[col_idx]:
                                fig = visualizer.create_attention_plot(
                                    attention_data, layer, head_idx
                                )
                                fig.update_layout(height=300, width=250)
                                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Statistics")
            cl1, cl2, cl3, cl4, cl5 = st.columns(5)
            with cl1:
                st.metric("Sequence Length", attention_data["seq_len"])
            with cl2:
                st.metric("Number of Layers", attention_data["num_layers"])
            with cl3:
                st.metric("Number of Heads", attention_data["num_heads"])

            if view_mode == "Single View":
                matrix = attention_data["attention"][layer][head]
                avg_attention = matrix.mean()
                max_attention = matrix.max()

                with cl4:
                    st.metric("Average Attention", f"{avg_attention:.3f}")
                with cl5:
                    st.metric("Max Attention", f"{max_attention:.3f}")


if __name__ == "__main__":
    main()
