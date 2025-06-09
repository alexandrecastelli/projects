import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_summary(path: str = "ff_summary.parquet") -> pd.DataFrame:
    """Load the aggregated financial data."""
    return pd.read_parquet(path)


def main() -> None:
    st.title("Fluxo Financeiro")
    st.markdown(
        "Este dashboard utiliza dados agregados a partir do arquivo `ff.parquet`.\n"
        "Use o script `preprocess_ff.py` para gerar o arquivo `ff_summary.parquet`\n"
        "a partir dos dados completos."
    )

    df = load_summary()

    insts = sorted(df["instituicao"].unique().tolist())
    inst_option = st.sidebar.multiselect("Instituição", insts, default=insts)

    filtro = df[df["instituicao"].isin(inst_option)]

    st.header("Valor total por tipo de movimentação")
    fig = px.bar(
        filtro,
        x="tipo_movimentacao",
        y="valor_total",
        color="instituicao",
        barmode="group",
        labels={"valor_total": "Valor Total"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.header("Valor total por tipo de contraparte")
    fig2 = px.bar(
        filtro,
        x="tipo_contraparte",
        y="valor_total",
        color="instituicao",
        barmode="group",
        labels={"valor_total": "Valor Total"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.header("Quantidade total por tipo de movimentação")
    fig3 = px.bar(
        filtro,
        x="tipo_movimentacao",
        y="quantidade_total",
        color="instituicao",
        barmode="group",
        labels={"quantidade_total": "Quantidade"},
    )
    st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
