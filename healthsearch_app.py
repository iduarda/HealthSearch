import re
import nltk
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# ⚠️ DEVE ser o primeiro comando do Streamlit — antes de qualquer st.*
st.set_page_config(page_title="HealthSearch", page_icon="🏥", layout="wide")


# ============================================
# FASE 1: Ingestão e Pré-processamento (Eduarda)
# ============================================
CORPUS = {
    "Doc 1": {
        "titulo": "Protocolo Emergência ECG",
        "texto": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos."
    },
    "Doc 2": {
        "titulo": "Guia de Farmacologia Cardíaca",
        "texto": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio."
    },
    "Doc 3": {
        "titulo": "Diretriz de Hipertensão Arterial",
        "texto": "A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI."
    },
    "Doc 4": {
        "titulo": "Manual de AVC Isquêmico",
        "texto": "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas."
    },
    "Doc 5": {
        "titulo": "Protocolo de Reanimação RCR",
        "texto": "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul."
    },
    "Doc 6": {
        "titulo": "Procedimentos de UTI Geral",
        "texto": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria."
    },
}

@st.cache_resource
def carregar_stopwords():
    nltk.download('stopwords', quiet=True)
    return set(stopwords.words('portuguese'))

STOPWORDS_PT = carregar_stopwords()

def preprocessar(texto: str) -> list[str]:
    texto = texto.lower()
    texto = re.sub(r'[^a-zà-ú0-9\s-]', ' ', texto)
    tokens = texto.split()
    tokens = [t for t in tokens if t not in STOPWORDS_PT]
    return tokens

corpus_tokenizado = {doc_id: preprocessar(doc["texto"]) for doc_id, doc in CORPUS.items()}
doc_ids = list(corpus_tokenizado.keys())


# ============================================
# FASE 2: Motor Léxico BM25 (Eduarda)
# ============================================
bm25 = BM25Okapi(list(corpus_tokenizado.values()))

def busca_bm25(query: str, k1: float, b: float) -> list[dict]:
    """Retorna os documentos ranqueados por relevância BM25."""
    bm25.k1 = k1
    bm25.b = b
    query_tokens = preprocessar(query)
    scores = bm25.get_scores(query_tokens)

    resultados = []
    for doc_id, score in zip(doc_ids, scores):
        resultados.append({
            "doc_id": doc_id,
            "titulo": CORPUS[doc_id]["titulo"],
            "texto": CORPUS[doc_id]["texto"],
            "score": round(score, 4)
        })
    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados


# ============================================
# FASE 3: Motor Semântico Vetorial (Milena)
# ============================================
MODELO_SEMANTICO = "paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource
def carregar_modelo_semantico():
    return SentenceTransformer(MODELO_SEMANTICO)

@st.cache_resource
def gerar_embeddings_corpus(_modelo):
    textos = [CORPUS[doc_id]["texto"] for doc_id in doc_ids]
    embeddings = _modelo.encode(textos, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings

modelo_semantico = carregar_modelo_semantico()
embeddings_corpus = gerar_embeddings_corpus(modelo_semantico)

def busca_semantica(query: str) -> list[dict]:
    """Retorna os documentos ranqueados por similaridade de cosseno."""
    embedding_query = modelo_semantico.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    similaridades = util.cos_sim(embedding_query, embeddings_corpus)[0].cpu().numpy()

    resultados = []
    for doc_id, score in zip(doc_ids, similaridades):
        resultados.append({
            "doc_id": doc_id,
            "titulo": CORPUS[doc_id]["titulo"],
            "texto": CORPUS[doc_id]["texto"],
            "score": round(float(score), 4)
        })
    resultados.sort(key=lambda x: x["score"], reverse=True)
    return resultados


# ============================================
# FASE 4: Fusão RRF (Anna)
# ============================================
K_RRF = 60  # constante de suavização de posição

def fusao_rrf(ranking_bm25: list[dict], ranking_semantico: list[dict], alfa: float) -> list[dict]:
    """Reciprocal Rank Fusion — combina os rankings BM25 e Semântico.

    Fórmula:
        Score_RRF(D) = α · [1 / (k + Rank_BM25)] + (1 − α) · [1 / (k + Rank_Sem)]

    Parâmetros
    ----------
    ranking_bm25 : lista já ordenada (melhor primeiro) vinda de busca_bm25()
    ranking_semantico : lista já ordenada vinda de busca_semantica()
    alfa : peso entre léxico (1.0) e semântico (0.0)
    """
    rank_bm25 = {r["doc_id"]: pos + 1 for pos, r in enumerate(ranking_bm25)}
    rank_sem  = {r["doc_id"]: pos + 1 for pos, r in enumerate(ranking_semantico)}

    resultados = []
    for doc_id in doc_ids:
        rb = rank_bm25.get(doc_id, len(doc_ids) + 1)
        rs = rank_sem.get(doc_id, len(doc_ids) + 1)

        score_rrf = alfa * (1 / (K_RRF + rb)) + (1 - alfa) * (1 / (K_RRF + rs))

        resultados.append({
            "doc_id": doc_id,
            "titulo": CORPUS[doc_id]["titulo"],
            "texto": CORPUS[doc_id]["texto"],
            "rank_bm25": rb,
            "rank_sem": rs,
            "score_rrf": round(score_rrf, 6)
        })

    resultados.sort(key=lambda x: x["score_rrf"], reverse=True)
    return resultados


def construir_matriz_comparativa(ranking_bm25, ranking_semantico, ranking_rrf):
    """Monta um DataFrame comparando as posições de cada doc nos 3 rankings."""
    rank_bm25 = {r["doc_id"]: pos + 1 for pos, r in enumerate(ranking_bm25)}
    rank_sem  = {r["doc_id"]: pos + 1 for pos, r in enumerate(ranking_semantico)}
    rank_rrf  = {r["doc_id"]: pos + 1 for pos, r in enumerate(ranking_rrf)}

    linhas = []
    for doc_id in doc_ids:
        linhas.append({
            "Documento": doc_id,
            "Título": CORPUS[doc_id]["titulo"],
            "Rank BM25": rank_bm25.get(doc_id, "—"),
            "Rank Semântico": rank_sem.get(doc_id, "—"),
            "Rank Híbrido (RRF)": rank_rrf.get(doc_id, "—"),
        })

    df = pd.DataFrame(linhas)
    df = df.sort_values("Rank Híbrido (RRF)")
    return df


# ============================================
# INTERFACE STREAMLIT (Anna — integração final)
# ============================================
st.title("🏥 HealthSearch")
st.caption("Motor de Busca Híbrido — BM25 + Semântico + Reciprocal Rank Fusion")

# ── Sidebar ──
st.sidebar.header("⚙️ Parâmetros")
query = st.sidebar.text_input("🔎 Busca:", placeholder="ex: infarto, ECG-12D, pressão alta…")

st.sidebar.markdown("---")
st.sidebar.subheader("BM25 (Léxico)")
k1 = st.sidebar.slider("k₁ — Saturação de Frequência", 0.0, 3.0, 1.2, 0.1)
b  = st.sidebar.slider("b — Normalização por Tamanho", 0.0, 1.0, 0.75, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("RRF (Fusão)")
alfa = st.sidebar.slider(
    "α — Peso Léxico ↔ Semântico",
    0.0, 1.0, 0.5, 0.05,
    help="α = 1.0 → 100 % BM25 · α = 0.0 → 100 % Semântico"
)

# ── Sem query → mostra corpus ──
if not query:
    st.info("Digite uma consulta na barra lateral para iniciar a busca.")
    with st.expander("📄 Corpus Médico (6 documentos)", expanded=False):
        for doc_id, doc in CORPUS.items():
            st.markdown(f"**{doc_id} — {doc['titulo']}**")
            st.write(doc["texto"])
            st.divider()
    st.stop()

# ── Executa as 3 buscas ──
resultados_bm25 = busca_bm25(query, k1, b)
resultados_sem  = busca_semantica(query)
resultados_rrf  = fusao_rrf(resultados_bm25, resultados_sem, alfa)

# ── Abas ──
tab_lex, tab_sem, tab_rrf, tab_matriz = st.tabs([
    "🔤 Léxico (BM25)",
    "🧠 Semântico",
    "⚡ Híbrido RRF",
    "📊 Matriz Comparativa",
])

# — Aba 1: Léxico (BM25) — Eduarda
with tab_lex:
    st.subheader("Ranking Léxico — BM25")
    st.caption(f"Parâmetros: k₁ = {k1}  ·  b = {b}")
    for i, r in enumerate(resultados_bm25):
        if i == 0 and r["score"] > 0:
            st.success(f"🥇 **{r['doc_id']} — {r['titulo']}**  ·  Score: `{r['score']}`")
        else:
            st.markdown(f"**{r['doc_id']} — {r['titulo']}**  ·  Score: `{r['score']}`")
        st.write(r["texto"])
        st.divider()

# — Aba 2: Semântico — Milena
with tab_sem:
    st.subheader("Ranking Semântico — Similaridade de Cosseno")
    st.caption(
        "Este motor recupera documentos relevantes mesmo sem correspondência "
        "exata de palavras — ex: buscar \"infarto\" também aproxima documentos "
        "sobre \"síndrome coronariana\"."
    )
    for i, r in enumerate(resultados_sem):
        if i == 0:
            st.success(f"🥇 **{r['doc_id']} — {r['titulo']}**  ·  Similaridade: `{r['score']}`")
        else:
            st.markdown(f"**{r['doc_id']} — {r['titulo']}**  ·  Similaridade: `{r['score']}`")
        st.write(r["texto"])
        st.divider()

# — Aba 3: Híbrido RRF — Anna
with tab_rrf:
    st.subheader("Ranking Híbrido — Reciprocal Rank Fusion")
    st.caption(f"α = {alfa}  ·  k_RRF = {K_RRF}")

    col_info1, col_info2 = st.columns(2)
    col_info1.metric("Peso Léxico (BM25)", f"{alfa:.0%}")
    col_info2.metric("Peso Semântico", f"{1 - alfa:.0%}")

    for i, r in enumerate(resultados_rrf):
        if i == 0:
            st.success(
                f"🥇 **{r['doc_id']} — {r['titulo']}**  ·  "
                f"Score RRF: `{r['score_rrf']}`  "
                f"(BM25: {r['rank_bm25']}º  ·  Sem: {r['rank_sem']}º)"
            )
        else:
            st.markdown(
                f"**{r['doc_id']} — {r['titulo']}**  ·  "
                f"Score RRF: `{r['score_rrf']}`  "
                f"(BM25: {r['rank_bm25']}º  ·  Sem: {r['rank_sem']}º)"
            )
        st.write(r["texto"])
        st.divider()

# — Aba 4: Matriz Comparativa — Anna
with tab_matriz:
    st.subheader("Matriz Comparativa de Rankings")
    st.caption("Posições de cada documento nos três métodos de busca (1 = mais relevante).")

    df_matriz = construir_matriz_comparativa(resultados_bm25, resultados_sem, resultados_rrf)

    st.dataframe(
        df_matriz,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank BM25": st.column_config.NumberColumn("Rank BM25", format="%dº"),
            "Rank Semântico": st.column_config.NumberColumn("Rank Semântico", format="%dº"),
            "Rank Híbrido (RRF)": st.column_config.NumberColumn("Rank Híbrido (RRF)", format="%dº"),
        }
    )

    # gráfico de barras comparativo
    st.markdown("---")
    st.subheader("Gráfico Comparativo de Posições")

    df_chart = df_matriz.set_index("Documento")[["Rank BM25", "Rank Semântico", "Rank Híbrido (RRF)"]]
    st.bar_chart(df_chart, horizontal=True)

    st.caption(
        "Barras menores = melhor posição no ranking. Compare como a fusão RRF "
        "equilibra os pontos fortes de cada motor."
    )
