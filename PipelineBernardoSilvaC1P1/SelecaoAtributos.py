
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

# ============================================================
# Configurações
# ============================================================
# Diretórios
DIRETORIO_PIPELINE = Path(__file__).resolve().parent
DIRETORIO_DATASETS = DIRETORIO_PIPELINE / "DatasetAtributos"

# Janelas temporais em relação ao TTOQ
JANELAS = [
    (0, 5),
    (0, 10),
    (0, 20),
]

# Parâmetros da seleção de atributos
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 500
TOP_N = 40

# Colunas que não participam da seleção de atributos
METADADOS = [
    "arquivo",
    "fase_faltosa",
    "barra",
]

# Atributos de interesse para a localização
ATRIBUTOS_DE_INTERESSE = [
    "soma_modulo_corrente_H1_maximo",
    "soma_modulo_corrente_H1_media",
    "soma_modulo_corrente_H1_minimo",
    "soma_modulo_tensao_H1_maximo",
    "soma_modulo_tensao_H1_media",
    "soma_modulo_tensao_H1_minimo",
]

# ============================================================
# Funções auxiliares
# ============================================================
def carregar_dataset(ciclos_antes, ciclos_depois):
    caminho = (
        DIRETORIO_DATASETS
        / f"atributos_{ciclos_antes}_{ciclos_depois}.parquet"
    )

    if not caminho.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {caminho}"
        )

    df = pd.read_parquet(caminho).copy()

    # Verifica a presença dos metadados necessários
    colunas_ausentes = set(METADADOS) - set(df.columns)

    if colunas_ausentes:
        raise ValueError(
            f"Metadados ausentes em {caminho.name}: "
            f"{sorted(colunas_ausentes)}"
        )

    # Verifica se existem simulações duplicadas
    if df["arquivo"].duplicated().any():
        raise ValueError(
            f"Arquivos duplicados em {caminho.name}."
        )

    return df

def selecionar_colunas_atributos(df):
    # Considera todos os atributos, incluindo os da fase faltosa
    colunas = [
        coluna
        for coluna in df.columns
        if coluna not in METADADOS
    ]

    if not colunas:
        raise ValueError(
            "Nenhum atributo disponível para seleção."
        )

    return colunas

def preparar_matriz(df, colunas):
    # Converte os atributos para valores numéricos
    X = df[colunas].apply(
        pd.to_numeric,
        errors="raise",
    )

    # Verifica a presença de valores inválidos
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Foram encontrados valores NaN ou infinitos nos atributos."
        )

    return X

def calcular_ranking_anova(X_treino, y_treino):
    # Remove atributos constantes no conjunto de treinamento
    variancias = X_treino.var(axis=0)

    colunas_variaveis = variancias[
        variancias > 0
    ].index.tolist()

    if not colunas_variaveis:
        raise ValueError(
            "Todos os atributos são constantes no treinamento."
        )

    # Calcula a estatística F e o valor-p de cada atributo
    valores_f, valores_p = f_classif(
        X_treino[colunas_variaveis],
        y_treino,
    )

    ranking = pd.DataFrame({
        "atributo": colunas_variaveis,
        "estatistica_f": valores_f,
        "valor_p": valores_p,
    })

    # Remove resultados inválidos do teste F
    ranking = ranking.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    ranking = ranking.dropna(
        subset=["estatistica_f"]
    )

    # Ordena os atributos pela estatística F decrescente
    ranking = ranking.sort_values(
        ["estatistica_f", "atributo"],
        ascending=[False, True],
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "posicao_anova",
        range(1, len(ranking) + 1),
    )

    return ranking

def calcular_ranking_rf(X_treino, y_treino):
    # Treina a Random Forest com todos os atributos disponíveis
    modelo = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    modelo.fit(
        X_treino,
        y_treino,
    )

    # Obtém a importância dos atributos pela redução de impureza
    ranking = pd.DataFrame({
        "atributo": X_treino.columns,
        "importancia_rf": modelo.feature_importances_,
    })

    # Ordena os atributos pela importância decrescente
    ranking = ranking.sort_values(
        ["importancia_rf", "atributo"],
        ascending=[False, True],
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "posicao_rf",
        range(1, len(ranking) + 1),
    )

    return ranking

# ============================================================
# Divisão dos conjuntos de treinamento e teste
# ============================================================
# Utiliza a primeira janela para definir um split fixo
if not JANELAS:
    raise ValueError(
        "A lista JANELAS está vazia."
    )

ciclos_antes_base, ciclos_depois_base = JANELAS[0]

df_base = carregar_dataset(
    ciclos_antes_base,
    ciclos_depois_base,
)

# Divide as simulações preservando a proporção de cada barra
arquivos_treino, arquivos_teste = train_test_split(
    df_base["arquivo"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_base["barra"],
)

arquivos_treino = set(arquivos_treino)
arquivos_teste = set(arquivos_teste)
arquivos_esperados = arquivos_treino.union(arquivos_teste)

# Verifica se existem simulações compartilhadas entre os conjuntos
if arquivos_treino.intersection(arquivos_teste):
    raise RuntimeError(
        "Há simulações compartilhadas entre treino e teste."
    )

print("=" * 100)
print("SELEÇÃO DE ATRIBUTOS - ANOVA E RANDOM FOREST")
print("=" * 100)
print(f"Treinamento: {len(arquivos_treino)} simulações")
print(f"Teste reservado: {len(arquivos_teste)} simulações")
print(f"Top N: {TOP_N} atributos")

# ============================================================
# Seleção de atributos por janela
# ============================================================
resultados_h1 = []
resumo_janelas = []

for ciclos_antes, ciclos_depois in JANELAS:
    nome_janela = f"{ciclos_antes}/{ciclos_depois}"

    print("\n" + "=" * 100)
    print(f"JANELA {nome_janela}")
    print("=" * 100)

    # --------------------------------------------------------
    # Leitura e preparação do dataset
    # --------------------------------------------------------
    df = carregar_dataset(
        ciclos_antes,
        ciclos_depois,
    )

    # Confirma que as janelas contêm as mesmas simulações
    if set(df["arquivo"]) != arquivos_esperados:
        raise ValueError(
            f"A janela {nome_janela} não contém as mesmas simulações."
        )

    # Seleciona todos os atributos disponíveis
    colunas_atributos = selecionar_colunas_atributos(df)

    # Verifica a presença dos seis atributos H1 de referência
    atributos_ausentes = set(ATRIBUTOS_DE_INTERESSE) - set(colunas_atributos)

    if atributos_ausentes:
        raise ValueError(
            f"Atributos H1 ausentes na janela {nome_janela}: "
            f"{sorted(atributos_ausentes)}"
        )

    X = preparar_matriz(
        df,
        colunas_atributos,
    )

    y = df["barra"].astype(int)

    # Seleciona apenas as simulações do conjunto de treinamento
    mascara_treino = df["arquivo"].isin(arquivos_treino)

    X_treino = X.loc[mascara_treino]
    y_treino = y.loc[mascara_treino]

    # --------------------------------------------------------
    # Ranking ANOVA
    # --------------------------------------------------------
    ranking_anova = calcular_ranking_anova(
        X_treino,
        y_treino,
    )

    top_anova = ranking_anova.head(TOP_N)

    # --------------------------------------------------------
    # Ranking Random Forest
    # --------------------------------------------------------
    ranking_rf = calcular_ranking_rf(
        X_treino,
        y_treino,
    )

    top_rf = ranking_rf.head(TOP_N)

    # --------------------------------------------------------
    # Comparação dos rankings
    # --------------------------------------------------------
    # Combina as posições e os valores dos dois métodos
    comparacao = ranking_anova.merge(
        ranking_rf,
        on="atributo",
        how="outer",
    )

    atributos_anova = set(top_anova["atributo"])
    atributos_rf = set(top_rf["atributo"])
    atributos_comuns = atributos_anova.intersection(atributos_rf)

    # Identifica os atributos presentes em cada Top 40
    comparacao["top_anova"] = comparacao["atributo"].isin(
        atributos_anova
    )

    comparacao["top_rf"] = comparacao["atributo"].isin(
        atributos_rf
    )

    comparacao["top_ambos"] = comparacao["atributo"].isin(
        atributos_comuns
    )

    # --------------------------------------------------------
    # Apresentação dos rankings gerais
    # --------------------------------------------------------
    print(f"\nAtributos avaliados: {len(colunas_atributos)}")

    print(f"\nTOP {TOP_N} - ANOVA")
    print(
        top_anova.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(f"\nTOP {TOP_N} - RANDOM FOREST")
    print(
        top_rf.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Atributos presentes nos dois rankings
    # --------------------------------------------------------
    comparacao_comuns = comparacao[
        comparacao["top_ambos"]
    ].copy()

    comparacao_comuns = comparacao_comuns[
        [
            "atributo",
            "posicao_anova",
            "posicao_rf",
        ]
    ].sort_values(
        ["posicao_anova", "posicao_rf"]
    )

    print(f"\nATRIBUTOS PRESENTES NOS DOIS TOP {TOP_N}")
    print(f"Total: {len(atributos_comuns)}")

    print(
        comparacao_comuns.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Ranking dos seis atributos H1
    # --------------------------------------------------------
    # Consulta as posições nos rankings completos
    comparacao_h1 = comparacao[
        comparacao["atributo"].isin(ATRIBUTOS_DE_INTERESSE)
    ].copy()

    comparacao_h1 = comparacao_h1[
        [
            "atributo",
            "posicao_anova",
            "estatistica_f",
            "posicao_rf",
            "importancia_rf",
            "top_anova",
            "top_rf",
        ]
    ]

    # Mantém a ordem dos atributos definida nas configurações
    comparacao_h1 = (
        comparacao_h1
        .set_index("atributo")
        .reindex(ATRIBUTOS_DE_INTERESSE)
        .reset_index()
    )

    print("\nRANKING DOS SEIS ATRIBUTOS H1")
    print(
        comparacao_h1.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # Armazena as posições para comparação entre as janelas
    for _, linha in comparacao_h1.iterrows():
        resultados_h1.append({
            "atributo": linha["atributo"],
            "janela": nome_janela,
            "posicao_anova": linha["posicao_anova"],
            "posicao_rf": linha["posicao_rf"],
        })

    # --------------------------------------------------------
    # Resumo da janela
    # --------------------------------------------------------
    resumo_janelas.append({
        "janela": nome_janela,
        "n_atributos": len(colunas_atributos),
        "n_top_anova": len(top_anova),
        "n_top_rf": len(top_rf),
        "n_atributos_comuns": len(atributos_comuns),
    })

# ============================================================
# Comparação dos atributos H1 entre todas as janelas
# ============================================================
df_h1 = pd.DataFrame(resultados_h1)

# Organiza as posições da ANOVA por janela
tabela_anova = df_h1.pivot(
    index="atributo",
    columns="janela",
    values="posicao_anova",
)

# Organiza as posições da Random Forest por janela
tabela_rf = df_h1.pivot(
    index="atributo",
    columns="janela",
    values="posicao_rf",
)

# Combina as posições dos dois métodos em uma única tabela
comparacao_final = pd.DataFrame(
    index=ATRIBUTOS_DE_INTERESSE
)

for ciclos_antes, ciclos_depois in JANELAS:
    janela = f"{ciclos_antes}/{ciclos_depois}"

    comparacao_final[f"ANOVA_{janela}"] = tabela_anova[janela]
    comparacao_final[f"RF_{janela}"] = tabela_rf[janela]

print("\n" + "=" * 130)
print("COMPARAÇÃO DOS SEIS ATRIBUTOS H1 ENTRE AS JANELAS")
print("=" * 130)

print(
    comparacao_final.to_string(
        na_rep="N/A"
    )
)

print("\nProcessamento concluído!")