import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
import seaborn as sns

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
df_orig = load_original(ORIG_CSV, usecols=['player_id','short_name','player_face_url','club_name','player_positions']) \
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
    display_names = choices.apply(lambda r: f"{r.short_name} — {r.club_name} ({int(r.player_id)})", axis=1)
    player_select = st.sidebar.selectbox("Jogador", options=choices['player_id'].tolist(), format_func=lambda pid: display_names[choices['player_id']==pid].values[0])
else:
    choices = df_prepared[['player_id','short_name']].drop_duplicates().sort_values('short_name')
    display_names = choices['short_name'].tolist()
    player_select = st.sidebar.selectbox("Jogador", options=choices['player_id'].tolist(), format_func=lambda pid: choices[choices['player_id']==pid]['short_name'].values[0])

model_choice = st.sidebar.selectbox("Modelo", ["RF_Basico", "RF_Hierarquico"])

run_button = st.sidebar.button("Rodar predição")

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
        # build feature vector for the player
        X_player = row_prep.iloc[0:1][features_used]

        # display header / photo / basic info
        st.title(f"{row_prep.iloc[0]['short_name']} — Previsão de posição")
        col1, col2 = st.columns([1,2])
        with col1:
            if row_orig is not None and pd.notna(row_orig.get('player_face_url', None)):
                st.image(row_orig['player_face_url'], width=180)
            else:
                st.write("Foto não disponível")
        with col2:
            st.markdown(f"**Club:** {row_prep.iloc[0].get('club_name','-')}")
            st.markdown(f"**Posições originais (dataset):** {row_prep.iloc[0].get('player_positions','-')}")
            st.markdown(f"**Overall / País:** {row_prep.iloc[0].get('overall','-')} / {row_prep.iloc[0].get('nationality_name','-')}")
            st.markdown(f"**Idade / Altura:** {row_prep.iloc[0].get('age','-')} / {row_prep.iloc[0].get('height_cm','-')}cm")

        # show feature table (top relevant features)
        st.subheader("Features usadas (input para o modelo)")
        st.table(X_player.T)

        # -------- RF Básico --------
        if model_choice == "RF_Basico":
            st.subheader("RF Básico — predição")
            pred_enc = rf_baseline.predict(X_player)[0]
            proba = rf_baseline.predict_proba(X_player)[0]

            # map classes to names using mapping from df_prepared (label_map_enc2name)
            classes_enc = list(rf_baseline.classes_)
            classes_names = [label_map_enc2name.get(int(e), str(int(e))) for e in classes_enc]

            # present prediction
            pred_name = label_map_enc2name.get(int(pred_enc), str(int(pred_enc)))
            st.markdown(f"**Predição (principal):** {pred_name}")

            # top-3 probabilities
            topk = 5 if len(proba) >= 5 else len(proba)
            idx = np.argsort(proba)[::-1][:topk]
            df_top = pd.DataFrame({
                "position": [classes_names[i] for i in idx],
                "probability": [float(proba[i]) for i in idx]
            })
            st.table(df_top)

            # feature importances (top 20)
            fi = rf_baseline.feature_importances_
            fi_series = pd.Series(fi, index=features_used).sort_values(ascending=False).head(20)
            st.subheader("Feature importances (top 20)")
            fig, ax = plt.subplots(figsize=(8,4))
            sns.barplot(x=fi_series.values, y=fi_series.index, ax=ax)
            ax.set_xlabel("Importance")
            st.pyplot(fig)

        # -------- RF Hierárquico --------
        else:
            st.subheader("RF Hierárquico — predição (macro → main)")
            # macro
            macro_pred = rf_macro.predict(X_player)[0]
            macro_proba = rf_macro.predict_proba(X_player)[0]
            st.markdown(f"**Macro predito:** {macro_name_map.get(int(macro_pred), str(int(macro_pred)))}")
            # show macro probs
            macro_classes = list(rf_macro.classes_)
            macro_names = [macro_name_map.get(int(e), str(int(e))) for e in macro_classes]
            df_macro_probs = pd.DataFrame({
                "macro": macro_names,
                "prob": [float(x) for x in macro_proba]
            }).sort_values("prob", ascending=False).reset_index(drop=True)
            st.table(df_macro_probs)

            # main: pick model according to macro_pred
            if int(macro_pred) in rf_main_models:
                rf_main = rf_main_models[int(macro_pred)]
                main_pred = rf_main.predict(X_player)[0]
                main_proba = rf_main.predict_proba(X_player)[0]

                # map classes via label_map_enc2name
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