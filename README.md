# 🏆 Previsão de Posição — EA FC 26

Classificação de posição de jogadores com Machine Learning

🔗 Aplicação online: https://previsao-posicao-fifa.streamlit.app/

## 📌 Sobre o Projeto

Este projeto tem como objetivo prever a posição principal de um jogador de futebol usando seus atributos do jogo EA FC 26.
A aplicação utiliza Random Forest em duas arquiteturas:

- Modelo Direto → prevê a posição final diretamente.

- Modelo Hierárquico (Macro → Main) → primeiro prevê o grupo de posição (Defesa, Meio, Ponta ou Ataque), depois prevê a posição específica.

## 👨‍💻 O usuário pode:

- Selecionar qualquer jogador do dataset (não inclui goleiros)
- Selecionar um modelo de previsão (Random Forest Direto ou Hierárquico)
- Visualizar seus atributos
- Ver previsões e probabilidades
- Ver gráficos de atributos
- Conferir a imagem, clube, idade, nacionalidade etc.

Os modelos são carregados automaticamente a partir do Hugging Face Hub usando um token seguro via Streamlit Secrets.

## ✨ Principais Funcionalidades
- 🔍 Previsão hierárquica de posição
- 🎯 Probabilidades ordenadas por confiança
- 🧩 Seleção de jogador diretamente pelo nome
- 🖼️ Foto oficial do jogador

## 🧠 Tecnologias Utilizadas

| Tecnologia        | Uso                                       |
|-------------------|-------------------------------------------|
| Python 3.10+      | Base do projeto                           |
| Pandas / NumPy    | Manipulação de dados                      |
| Scikit-Learn      | Treinamento dos modelos de classificação  |
| Joblib            | Serialização dos modelos                  |
| Streamlit         | Interface web interativa                  |
| Plotly            | Gráficos em estilo radar/pizza            |
| Hugging Face Hub  | Armazenamento dos modelos                 |
| Git + GitHub      | Versionamento e deploy automático         |
