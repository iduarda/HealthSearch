# HealthSearch — Motor de Busca Híbrido (BM25 + Semântico)

Projeto desenvolvido para a disciplina **Tendências em Ciência da Computação** — UNIPÊ, como parte do **Laboratório Prático 05 — Desafio Integrador HealthSearch**.

O HealthSearch é um motor de busca sobre um corpus médico, que combina **recuperação léxica (BM25)** e **recuperação semântica (embeddings vetoriais)** através de **Reciprocal Rank Fusion (RRF)**, expondo os três rankings lado a lado em uma interface interativa construída com Streamlit.

## Equipe

| Aluna | Responsabilidade |
|---|---|
| **Maria Eduarda** | Fase 1 (Ingestão e Pré-processamento) e Fase 2 (Motor Léxico BM25) |
| **Milena Azevêdo** | Fase 3 (Motor Semântico Vetorial) |
| **Anna Maria** | Fase 4 (Fusão RRF), Matriz Comparativa e integração final do `healthsearch_app.py` |

## Arquitetura da Solução

O pipeline segue quatro etapas sequenciais, integradas em um único arquivo (`healthsearch_app.py`):

```
Corpus (6 docs) → Pré-processamento → ┬─ BM25 (Léxico) ─┐
                                       └─ Embeddings ────┴─ RRF (Híbrido) → Interface Streamlit
                                          (Semântico)
```

1. **Ingestão e Pré-processamento** — carregamento estático do corpus e limpeza de texto (minúsculas, remoção de caracteres especiais preservando acentos e códigos técnicos, remoção de stopwords em português via NLTK).
2. **Motor Léxico (BM25)** — ranking via `rank_bm25`, com parâmetros `k1` e `b` ajustáveis em tempo real.
3. **Motor Semântico** — embeddings gerados com `sentence-transformers` (modelo `paraphrase-multilingual-MiniLM-L12-v2`) e similaridade de cosseno.
4. **Fusão RRF** — combinação dos dois rankings pela fórmula `Score(D) = α·[1/(k+RankBM25)] + (1−α)·[1/(k+RankSemântico)]`, com `k=60` e peso `α` ajustável.

O relatório técnico completo, com detalhes de implementação e testes de cada fase, está em [`docs/Relatorio_HealthSearch_Final.pdf`](Relatorio_HealthSearch.pdf).

## Interface

A aplicação expõe quatro abas:
- **Léxico (BM25)**
- **Semântico**
- **Híbrido RRF**
- **Matriz Comparativa** — gráfico comparando a posição de cada documento nos três rankings

## Como executar

**Pré-requisitos:** Python 3.10+

```bash
# 1. Clone o repositório
git clone https://github.com/<seu-usuario>/healthsearch.git
cd healthsearch

# 2. Crie e ative um ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a aplicação
streamlit run healthsearch_app.py
```

A aplicação abrirá automaticamente em `http://localhost:8501`.

## Tecnologias

- **Python 3**
- **Streamlit** — interface web interativa
- **rank_bm25** — ranking léxico (BM25Okapi)
- **sentence-transformers** — embeddings semânticos multilíngues
- **NLTK** — stopwords em português
- **Pandas** — construção da matriz comparativa

## Estrutura do repositório

```
healthsearch/
├── README.md
├── requirements.txt
├── .gitignore
├── healthsearch_app.py
├── docs/
│   └── Relatorio_HealthSearch_Final.pdf
└── screenshots/
    ├── aba_lexico.png
    ├── aba_semantico.png
    ├── aba_hibrido_rrf.png
    └── matriz_comparativa.png
```

## Licença

Projeto acadêmico desenvolvido para fins educacionais — UNIPÊ, 2026.
