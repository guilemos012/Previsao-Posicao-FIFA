import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
import plotly.graph_objects as go
import requests
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import tempfile

st.set_page_config(layout="wide", page_title="Previsão de Posição - EA FC26")

HF_REPO_ID = "guilemos012/previsao-posicao-models"
MODEL_DIR = "Treino/models_comprimido"
os.makedirs(MODEL_DIR, exist_ok=True)

MODELS = [
    "rf_best_model_baseline.pkl",
    "rf_macro_tuned.pkl",
    "rf_main_enc_0.pkl",
    "rf_main_enc_1.pkl",
    "rf_main_enc_2.pkl",
    "rf_main_enc_3.pkl",
]

PREPARED_CSV = "Análise/players_prepared.csv"   
ORIG_CSV     = "FC26.csv"                       

@st.cache_data
def load_prepared(path=PREPARED_CSV):
    return pd.read_csv(path)

@st.cache_data
def load_original(path=ORIG_CSV, usecols=None):
    return pd.read_csv(path, usecols=usecols)

def ensure_models(repo_id=HF_REPO_ID, model_fnames=MODELS, token=None):
    """
    Garante que os arquivos listados em model_fnames existam em MODEL_DIR.
    Se não existirem, baixa do HF Hub usando hf_hub_download.
    Retorna dict filename -> local_path
    """
    local_paths = {}
    for fname in model_fnames:
        out_path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(out_path):
            hf_hub_download(repo_id=repo_id, filename=fname, local_dir=MODEL_DIR, token=token)
        local_paths[fname] = out_path
    return local_paths

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Inicialização: carregar dados principais e garantir modelos
st.sidebar.title("Configurações")

# carregar CSVs
df_prepared = load_prepared(PREPARED_CSV)
df_orig = load_original(ORIG_CSV, usecols=['player_id','short_name','player_face_url','club_name','player_positions', 'nationality_name']) \
          if Path(ORIG_CSV).exists() else pd.DataFrame()

# procura por player_id
df_orig_index = df_orig.set_index('player_id') if not df_orig.empty else None

hf_token = st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else None

with st.spinner("Verificando modelos..."):
    model_paths = ensure_models(token=hf_token)

# carregar modelos
rf_baseline = load_model(model_paths["rf_best_model_baseline.pkl"])
rf_macro = load_model(model_paths["rf_macro_tuned.pkl"])
rf_main_models = {i: load_model(model_paths[f"rf_main_enc_{i}.pkl"]) for i in range(4)}

# Construir mapeamentos de position enc -> name a partir do prepared df
if 'main_position_enc' in df_prepared.columns and 'main_position' in df_prepared.columns:
    mapping_df = df_prepared[['main_position_enc','main_position']].drop_duplicates().sort_values('main_position_enc')
    label_map_enc2name = dict(zip(mapping_df['main_position_enc'], mapping_df['main_position']))
else:
    label_map_enc2name = {int(e): str(e) for e in rf_baseline.classes_}

macro_name_map = {0: "Defender", 1: "Midfielder", 2: "Striker", 3: "Winger"}

# carregar lista de features que o modelo espera (arquivo JSON ou inferir)
FEATURES_JSON = "Treino/results_basico/features_used_v1.json"
if Path(FEATURES_JSON).exists():
    with open(FEATURES_JSON, 'r', encoding='utf-8') as f:
        features_used = json.load(f)
else:
    exclude = {'player_id','main_position','main_position_enc','macro_position','macro_position_enc','secondary_position','secondary_position_enc'}
    features_used = [c for c in df_prepared.columns if c not in exclude and df_prepared[c].dtype in [np.int64, np.float64, 'int64','float64']]

st.sidebar.markdown("### Escolha jogador e modelo")
if not df_orig.empty:
    choices = df_orig[['player_id','short_name','club_name']].drop_duplicates().sort_values('short_name')
    display_names = choices.apply(lambda r: f"{r.short_name} — {r.club_name}", axis=1)
    player_select = st.sidebar.selectbox("Jogador", options=choices['player_id'].tolist(), format_func=lambda pid: display_names[choices['player_id']==pid].values[0])
else:
    choices = df_prepared[['player_id','short_name']].drop_duplicates().sort_values('short_name')
    display_names = choices['short_name'].tolist()
    player_select = st.sidebar.selectbox("Jogador", options=choices['player_id'].tolist(), format_func=lambda pid: choices[choices['player_id']==pid]['short_name'].values[0])

model_choice = st.sidebar.selectbox("Modelo", ["Random Forest Direto", "Random Forest Hierárquico"])

run_button = st.sidebar.button("Rodar predição")


# Funções pra gerar gráfico com base nos atributos principais
def get_macro_values(row):
    """
    row: pandas Series com colunas:
     'pace','shooting','passing','dribbling','defending','physic'
    retorna lista na ordem [pace, shooting, passing, dribbling, defending, physic]
    Garantindo valores 0..100
    """
    keys = ['pace','shooting','passing','dribbling','defending','physic']
    vals = []
    for k in keys:
        v = row.get(k, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            vals.append(0.0)
        else:
            try:
                vals.append(float(v))
            except:
                vals.append(0.0)
    # clip 0..100
    vals = [max(0.0, min(100.0, x)) for x in vals]
    return vals

def plot_pizza_plotly(values, title="Atributos: (0-100)"):
    """
    values: array com 6 valores (0..100)
    retorna figure plotly
    """
    labels = ["Pace","Shooting","Passing","Dribbling","Defending","Physic"]
    thetas = [0, 60, 120, 180, 240, 300]
    fig = go.Figure()

    fig.add_trace(go.Barpolar(
        r = values,
        theta = thetas,
        width = [60]*6,
        name = "Stats",
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.9,
        hovertemplate = "%{theta}°<br>%{r:.0f}"
    ))

    fig.update_layout(
        template=None,
        polar = dict(
            radialaxis = dict(range=[0,100], showticklabels=False, ticks="", showline=False, gridcolor='lightgray'),
            angularaxis = dict(tickmode='array', tickvals=thetas, ticktext=labels, rotation=90, direction="clockwise")
        ),
        showlegend=False,
        title = dict(text=title, x=0.5)
    )

    return fig

# Função pra exibir a foto do jogador (pela url estava dando problema)
@st.cache_data(show_spinner=False)
def fetch_image_bytes(url, timeout=8):
    """
    Tenta baixar a imagem e retorna bytes.
    Usa um user-agent para aumentar chance de sucesso.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        # checar content-type simples
        ctype = resp.headers.get("Content-Type", "")
        if not ctype.startswith("image"):
            return None
        return resp.content
    except Exception as e:
        return None

def display_player_image(row_orig, width=180):
    """
    row_orig: Series com chave 'player_face_url' (string)
    tenta st.image(url) primeiro; se não renderizar, baixa e mostra via bytes.
    """
    if row_orig is None:
        st.write("Foto não disponível")
        return

    url = row_orig.get("player_face_url", None)
    if not url or not isinstance(url, str):
        st.write("Foto não disponível")
        return

    img_bytes = fetch_image_bytes(url)
    if img_bytes:
        try:
            img = Image.open(BytesIO(img_bytes))
            st.image(img, width=width)
            return
        except Exception:
            st.write("Não foi possível renderizar a imagem a partir dos bytes.")
            return

    # falha geral
    st.write("Foto não disponível (falha ao baixar).")

# Quando clicar: rodar inferência e exibir info
def get_player_rows(pid):
    pid = int(pid)
    row_prep = df_prepared[df_prepared['player_id']==pid]
    row_orig = df_orig_index.loc[pid] if (df_orig_index is not None and pid in df_orig_index.index) else None
    return row_prep, row_orig

if run_button:
    pid = int(player_select)
    row_prep, row_orig = get_player_rows(pid)

    if row_prep.shape[0] == 0:
        st.error("Jogador não encontrado em players_prepared.csv.")
    else:
        X_player = row_prep.iloc[0:1][features_used]

        # informações básicas
        st.title(f"{row_prep.iloc[0]['short_name']} — Previsão de Posição")
        col1, col2 = st.columns([1,2])
        with col1:
            display_player_image(row_orig, width=180)

        with col2:
            prep = row_prep.iloc[0]

            st.markdown(f"**Clube:** {row_orig.get('club_name', '-') if row_orig is not None else '-'}")
            st.markdown(f"**Posições Originais (dataset):** {row_orig.get('player_positions','-') if row_orig is not None else '-'}")
            st.markdown(f"**Overall:** {prep.get('overall','-')}")
            st.markdown(f"**País:** {row_orig.get('nationality_name', '-') if row_orig is not None else '-'}")
            st.markdown(f"**Idade:** {prep.get('age','-')}")

        # printar gráfico com atributos
        values = get_macro_values(row_prep)
        fig = plot_pizza_plotly(values, title=f"Atributos")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Features usadas (input para o modelo)")
        st.table(X_player.T)

        # -------- RF Básico --------
        if model_choice == "Random Forest Direto":
            st.subheader("Random Forest Direto — predição")
            pred_enc = rf_baseline.predict(X_player)[0]
            proba = rf_baseline.predict_proba(X_player)[0]

            classes_enc = list(rf_baseline.classes_)
            classes_names = [label_map_enc2name.get(int(e), str(int(e))) for e in classes_enc]

            pred_name = label_map_enc2name.get(int(pred_enc), str(int(pred_enc)))
            st.markdown(f"**Predição (principal):** {pred_name}")

            topk = 5 if len(proba) >= 5 else len(proba)
            idx = np.argsort(proba)[::-1][:topk]
            df_top = pd.DataFrame({
                "position": [classes_names[i] for i in idx],
                "probability": [float(proba[i]) for i in idx]
            })
            st.table(df_top)

        # -------- RF Hierárquico --------
        else:
            st.subheader("Random Forest Hierárquico — predição (macro → main)")
            # macro
            macro_pred = rf_macro.predict(X_player)[0]
            macro_proba = rf_macro.predict_proba(X_player)[0]
            st.markdown(f"**Macro predito:** {macro_name_map.get(int(macro_pred), str(int(macro_pred)))}")

            macro_classes = list(rf_macro.classes_)
            macro_names = [macro_name_map.get(int(e), str(int(e))) for e in macro_classes]
            df_macro_probs = pd.DataFrame({
                "macro": macro_names,
                "prob": [float(x) for x in macro_proba]
            }).sort_values("prob", ascending=False).reset_index(drop=True)
            st.table(df_macro_probs)

            # main: escolher model de acordo com o macro_position
            if int(macro_pred) in rf_main_models:
                rf_main = rf_main_models[int(macro_pred)]
                main_pred = rf_main.predict(X_player)[0]
                main_proba = rf_main.predict_proba(X_player)[0]

                classes_enc_main = list(rf_main.classes_)
                classes_names_main = [label_map_enc2name.get(int(e), str(int(e))) for e in classes_enc_main]

                # present main prediction
                st.markdown(f"**Main predito (condicionado ao macro):** {label_map_enc2name.get(int(main_pred), str(int(main_pred)))}")

                # show top-k main probs
                topk = 5 if len(main_proba) >= 5 else len(main_proba)
                idx = np.argsort(main_proba)[::-1][:topk]
                df_top_main = pd.DataFrame({
                    "position": [classes_names_main[i] for i in idx],
                    "probability": [float(main_proba[i]) for i in idx]
                })
                st.table(df_top_main)
            else:
                st.warning("Modelo main para o macro predito não foi encontrado.")

        st.success("Predição concluída.")