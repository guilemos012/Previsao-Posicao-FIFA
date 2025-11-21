# 🏆 Previsão de Posição — EA FC 26

Aplicação que prediz a posição ideal de um jogador de FIFA (EA FC 26) utilizando técinas de Ciência de Dados e Machine Learning.
O modelo recebe os atributos do jogador e retorna a posição mais provável, além de probabilidades detalhadas e um gráfico com os atributos principais.

🔗 Acesse a aplicação online:
👉 https://previsao-posicao-fifa.streamlit.app/

## 📌 Sobre o projeto
A aplicação utiliza Random Forest em duas arquiteturas:

- Modelo Direto → prevê a posição final diretamente.
- Modelo Hierárquico (Macro → Main) → primeiro prevê o grupo de posição (Defesa, Meio, Ponta ou Ataque), depois prevê a posição específica.

## 🧠 Visão Geral da Metodologia
O processo foi dividido em etapas bem definidas:

### 1️⃣ Coleta e Organização dos Dados
Foram utilizados dois arquivos principais:

| Arquivo	                 | Conteúdo
|--------------------------|-----------------------------------------------------------------------------------|
|FC26.csv	                 |  Dados originais do FIFA/FC26: nome, clube, idade, atributos técnicos, foto etc.  |
|players_prepared.csv	     |  Versão tratada para modelagem: features processadas e filtradas.                 |

### 2️⃣ Pré-Processamento
Principais etapas:

- Remoção de goleiros.
- Seleção de atributos relevantes.
- Normalização/ajuste de tipos.
- Criação da variável-alvo (posição final).
- Remoção de colunas irrelevantes para o modelo (nome, id, foto etc.).

### 3️⃣ Engenharia de Atributos (Feature Engineering)
Foram criadas diversas features compostas para representar melhor o estilo do jogador:

Exemplos de features:
- feat_offensive_index
- feat_defensive_index
- feat_speed_index
- feat_stamina_strength_ratio
- feat_attack_defense_ratio
- feat_aerial_ability
- feat_power_index
- feat_lateral_score, feat_winger_score, feat_striker_score, etc.

Essas features auxiliaram bastante na separação entre posições próximas (ex: LB vs RB, CM vs CDM, RM vs RW).

### 4️⃣ Modelagem
Testaram-se dois tipos principais de modelo:

#### A) Random Forest Direto (modelo baseline)
Classificador único que recebe todas as features e tenta predizer diretamente a posição final (CB, CM, ST…).
- ✔️ Fácil de treinar
- ✔️ Resultado satisfatório em casos fáceis
- ❌ Dificuldade em posições semelhantes (ex: CM ↔ CDM)

#### B) Random Forest Hierárquico (macro → main)
Abordagem em dois níveis:
1. Macroclassificação (Defender / Midfielder / Striker / Winger)
2. Modelo especializado para cada macro-posição

Exemplo:
```
macro_pred = "DEF"
→ carrega modelo rf_main_enc_DEF.pkl
→ prediz entre {CB, LB, RB}
```

- ✔️ Aumenta precisão em casos difíceis
- ✔️ Explica melhor o comportamento do jogador
- ✔️ Melhora a interpretabilidade

### 5️⃣ Treino, Teste e Avaliação
Divisão: 80% treino / 20% teste

Avaliação com:
- F1-score
- Balanced accuracy
- Confusion matrix
- Probabilidades por classe

#### 📊 Principais resultados:

Random Forest Direto:
- F1-score macro: ~0.73
- Melhor para posições claras: ST, CB

Hierárquico:
- F1-score macro: ~0.76
- Aumenta acertos em posições parecidas (CM/CDM, LB/RB)
- Diminui erros graves

## 🌐 Deploy
O deploy foi feito com:
- Streamlit Community Cloud para hospedar a UI.
- HuggingFace Hub para armazenar os modelos.
- GitHub para versionamento e build automático.
- O app baixa os modelos automaticamente no primeiro uso.

## 🛠️ Tecnologias Utilizadas

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

## 📝 Conclusões
- O modelo se mostrou consistente e generaliza bem.
- A abordagem hierárquica tem performance superior.
- Feature engineering foi fundamental para separar posições semelhantes.
- A interface Streamlit tornou o projeto acessível e reproduzível.
