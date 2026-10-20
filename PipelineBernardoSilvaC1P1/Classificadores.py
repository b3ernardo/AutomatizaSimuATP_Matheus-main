
import numpy as np
import pandas as pd
import optuna
import json
import hashlib
import os
import sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from Regionalizacoes import REGIONALIZACOES


# ============================================================
# Configurações
# ============================================================
# Diretórios
DIRETORIO_PIPELINE = Path(__file__).resolve().parent
DIRETORIO_DATASETS = DIRETORIO_PIPELINE / "DatasetAtributos16"
CAMINHO_CACHE_RF = DIRETORIO_PIPELINE / "hiperparametros_rf_16.json"

# Janelas temporais em relação ao TTOQ
JANELAS = [
    (0, 5),
    (0, 10),
    (0, 20),
]

# Regionalizações utilizadas na classificação
NUMEROS_REGIOES = list(range(2, 34))

# Atributos utilizados na classificação
ATRIBUTOS_DE_INTERESSE = [
    "soma_modulo_corrente_H1_maximo",
    "soma_modulo_corrente_H1_media",
    "soma_modulo_corrente_H1_minimo",
    "soma_modulo_tensao_H1_maximo",
    "soma_modulo_tensao_H1_media",
    "soma_modulo_tensao_H1_minimo",
]

# Parâmetros da classificação
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 500
N_NEIGHBORS = 5
N_TRIALS = 20
N_FOLDS = 5

# Colunas necessárias para identificar as simulações
METADADOS = [
    "arquivo",
    "fase_faltosa",
    "barra",
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

    # Verifica a presença das colunas necessárias
    colunas_necessarias = METADADOS + ATRIBUTOS_DE_INTERESSE
    colunas_ausentes = set(colunas_necessarias) - set(df.columns)

    if colunas_ausentes:
        raise ValueError(
            f"Colunas ausentes em {caminho.name}: "
            f"{sorted(colunas_ausentes)}"
        )

    # Verifica se existem simulações duplicadas
    if df["arquivo"].duplicated().any():
        raise ValueError(
            f"Arquivos duplicados em {caminho.name}."
        )

    return df


def preparar_matriz(df):
    # Seleciona exclusivamente os seis atributos de interesse
    X = df[ATRIBUTOS_DE_INTERESSE].apply(
        pd.to_numeric,
        errors="raise",
    )

    # Verifica a presença de valores inválidos
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Foram encontrados valores NaN ou infinitos nos atributos."
        )

    return X


def construir_mapa_barra_regiao(regioes):
    # Associa cada barra à região correspondente
    mapa = {}

    for nome_regiao, barras in regioes.items():
        for barra in barras:
            if barra in mapa:
                raise ValueError(
                    f"A barra {barra} pertence a mais de uma região."
                )

            mapa[barra] = nome_regiao

    return mapa


def validar_regionalizacoes(barras_esperadas):
    # Verifica se cada regionalização contém todas as barras
    for numero_regioes in NUMEROS_REGIOES:
        if numero_regioes not in REGIONALIZACOES:
            raise ValueError(
                f"Regionalização {numero_regioes} não encontrada."
            )

        regioes = REGIONALIZACOES[numero_regioes]

        if len(regioes) != numero_regioes:
            raise ValueError(
                f"Regionalização {numero_regioes}: "
                f"quantidade incorreta de regiões."
            )

        mapa = construir_mapa_barra_regiao(regioes)

        if set(mapa) != barras_esperadas:
            raise ValueError(
                f"Regionalização {numero_regioes}: "
                "conjunto de barras incompatível com o dataset."
            )


def criar_alvo_regiao(df, numero_regioes):
    # Converte a barra da falta no rótulo da região
    regioes = REGIONALIZACOES[numero_regioes]
    mapa = construir_mapa_barra_regiao(regioes)

    y = df["barra"].astype(int).map(mapa)

    if y.isna().any():
        raise ValueError(
            f"Existem barras sem região na regionalização "
            f"{numero_regioes}."
        )

    return y


def criar_classificadores():
    # Cria novos modelos para cada experimento
    # O escalonamento é ajustado somente no treinamento
    classificadores = {
        "Random Forest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=N_NEIGHBORS,
            ),
        ),

        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
            ),
        ),

        "Regressão Logística": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    }

    return classificadores


def otimizar_random_forest(X_treino, y_treino):
    # Otimiza os hiperparâmetros somente com os dados de treinamento
    validacao = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    def objetivo(trial):
        parametros = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }

        modelo = RandomForestClassifier(
            **parametros,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        pontuacoes = cross_val_score(
            modelo,
            X_treino,
            y_treino,
            cv=validacao,
            scoring="f1_macro",
            n_jobs=1,
        )

        return float(np.mean(pontuacoes))

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    estudo = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    estudo.optimize(objetivo, n_trials=N_TRIALS)

    return estudo.best_params, estudo.best_value


# ============================================================
# Cache dos hiperparâmetros da Random Forest
# ============================================================
def assinatura_otimizacao(X_treino, y_treino):
    # Impede reutilização de parâmetros após alterações nos dados
    # ou nas configurações da otimização.
    resumo = {
        "atributos": ATRIBUTOS_DE_INTERESSE,
        "n_trials": N_TRIALS,
        "n_folds": N_FOLDS,
        "random_state": RANDOM_STATE,
        "sklearn": sklearn.__version__,
        "optuna": optuna.__version__,
        "busca": {
            "n_estimators": [100, 600, 100],
            "max_depth": [None, 5, 10, 20, 40],
            "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 5],
            "max_features": ["sqrt", "log2", None],
        },
    }
    assinatura = hashlib.sha256(
        json.dumps(resumo, sort_keys=True).encode("utf-8")
    )
    assinatura.update(pd.util.hash_pandas_object(X_treino, index=True).values.tobytes())
    assinatura.update(pd.util.hash_pandas_object(y_treino, index=True).values.tobytes())
    return assinatura.hexdigest()


def carregar_cache_rf():
    if not CAMINHO_CACHE_RF.is_file():
        raise FileNotFoundError(
            f"Arquivo de hiperparâmetros não encontrado: {CAMINHO_CACHE_RF}"
        )

    with CAMINHO_CACHE_RF.open("r", encoding="utf-8") as arquivo:
        cache = json.load(arquivo)

    if not isinstance(cache, dict):
        raise ValueError("Formato inválido do cache de hiperparâmetros.")

    return cache


def salvar_cache_rf(cache):
    # Substituição atômica: preserva os resultados anteriores em caso
    # de interrupção durante a escrita do arquivo.
    caminho_temporario = CAMINHO_CACHE_RF.with_suffix(".json.tmp")
    try:
        with caminho_temporario.open("w", encoding="utf-8") as arquivo:
            json.dump(cache, arquivo, ensure_ascii=False, indent=4)
        os.replace(caminho_temporario, CAMINHO_CACHE_RF)
    finally:
        if caminho_temporario.exists():
            caminho_temporario.unlink()


def obter_hiperparametros_rf(X_treino, y_treino, nome_janela, numero_regioes, cache):
    chave = f"{nome_janela}|{numero_regioes}"
    registro = cache.get(chave)

    if not isinstance(registro, dict) or not isinstance(registro.get("parametros"), dict):
        raise KeyError(
            f"Hiperparâmetros não encontrados no JSON para a janela "
            f"{nome_janela} e {numero_regioes} regiões (chave: {chave})."
        )

    assinatura = assinatura_otimizacao(X_treino, y_treino)
    if registro.get("assinatura") != assinatura:
        raise ValueError(
            f"Os hiperparâmetros salvos para a janela {nome_janela} e "
            f"{numero_regioes} regiões não correspondem aos dados ou "
            f"às configurações atuais. Refaça a otimização para essa configuração."
        )

    print(
        f"[RF] Janela {nome_janela} | {numero_regioes} regiões | "
        "hiperparâmetros carregados do JSON",
        flush=True,
    )
    return registro["parametros"], registro.get("f1_validacao")


# ============================================================
# Divisão dos conjuntos de treinamento e teste
# ============================================================
# Utiliza a primeira janela para definir um split fixo
if not JANELAS:
    raise ValueError(
        "A lista JANELAS está vazia."
    )

if not NUMEROS_REGIOES:
    raise ValueError(
        "A lista NUMEROS_REGIOES está vazia."
    )

ciclos_antes_base, ciclos_depois_base = JANELAS[0]

df_base = carregar_dataset(
    ciclos_antes_base,
    ciclos_depois_base,
)

# Valida as regionalizações com base nas barras disponíveis
barras_esperadas = set(
    df_base["barra"].astype(int)
)

validar_regionalizacoes(barras_esperadas)

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
print("CLASSIFICAÇÃO DE FALTAS DE ALTA IMPEDÂNCIA")
print("=" * 100)
print(f"Treinamento: {len(arquivos_treino)} simulações")
print(f"Teste: {len(arquivos_teste)} simulações")
print(f"Atributos: {len(ATRIBUTOS_DE_INTERESSE)}")
print(f"Regionalizações: {len(NUMEROS_REGIOES)}")


# ============================================================
# Classificação por janela e regionalização
# ============================================================
resultados = []
cache_rf = carregar_cache_rf()

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

    # Confirma que todas as janelas contêm as mesmas simulações
    if set(df["arquivo"]) != arquivos_esperados:
        raise ValueError(
            f"A janela {nome_janela} não contém as mesmas simulações."
        )

    # Confirma que as barras correspondem às do dataset inicial
    if set(df["barra"].astype(int)) != barras_esperadas:
        raise ValueError(
            f"A janela {nome_janela} contém barras incompatíveis."
        )

    X = preparar_matriz(df)

    mascara_treino = df["arquivo"].isin(arquivos_treino)
    mascara_teste = df["arquivo"].isin(arquivos_teste)

    X_treino = X.loc[mascara_treino]
    X_teste = X.loc[mascara_teste]

    # --------------------------------------------------------
    # Classificação por número de regiões
    # --------------------------------------------------------
    for numero_regioes in NUMEROS_REGIOES:
        # Converte as barras nos rótulos da regionalização
        y = criar_alvo_regiao(
            df,
            numero_regioes,
        )

        y_treino = y.loc[mascara_treino]
        y_teste = y.loc[mascara_teste]

        # Verifica se todas as regiões aparecem no treinamento
        if y_treino.nunique() != numero_regioes:
            raise ValueError(
                f"Regionalização {numero_regioes}: "
                "existem regiões ausentes no treinamento."
            )

        # ----------------------------------------------------
        # Treinamento e avaliação dos classificadores
        # ----------------------------------------------------
        classificadores = criar_classificadores()

        for nome_classificador, modelo in classificadores.items():
            if nome_classificador == "Random Forest":
                melhores_parametros, f1_validacao = obter_hiperparametros_rf(
                    X_treino,
                    y_treino,
                    nome_janela,
                    numero_regioes,
                    cache_rf,
                )
                modelo.set_params(**melhores_parametros)

            modelo.fit(
                X_treino,
                y_treino,
            )

            y_predito = modelo.predict(
                X_teste
            )

            # Calcula as métricas no conjunto de teste
            acuracia = accuracy_score(
                y_teste,
                y_predito,
            )

            f1_macro = f1_score(
                y_teste,
                y_predito,
                average="macro",
                zero_division=0,
            )

            resultados.append({
                "janela": nome_janela,
                "n_regioes": numero_regioes,
                "classificador": nome_classificador,
                "acuracia": acuracia,
                "f1_macro": f1_macro,
            })

    # --------------------------------------------------------
    # Resultados da janela
    # --------------------------------------------------------
    df_janela = pd.DataFrame(
        resultados
    )

    df_janela = df_janela[
        df_janela["janela"] == nome_janela
    ]

    # Organiza a acurácia por número de regiões e classificador
    tabela_acuracia = df_janela.pivot(
        index="n_regioes",
        columns="classificador",
        values="acuracia",
    )

    # Organiza o Macro-F1 da mesma maneira
    tabela_f1 = df_janela.pivot(
        index="n_regioes",
        columns="classificador",
        values="f1_macro",
    )

    print("\nACURÁCIA POR NÚMERO DE REGIÕES")
    print(
        tabela_acuracia.to_string(
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nMACRO-F1 POR NÚMERO DE REGIÕES")
    print(
        tabela_f1.to_string(
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ============================================================
# Comparação dos classificadores entre as janelas
# ============================================================
df_resultados = pd.DataFrame(
    resultados
)

# Seleciona os resultados da classificação individual das barras
numero_maximo_regioes = max(NUMEROS_REGIOES)

df_barras = df_resultados[
    df_resultados["n_regioes"] == numero_maximo_regioes
].copy()

# Organiza a acurácia dos classificadores por janela
comparacao_acuracia = df_barras.pivot(
    index="classificador",
    columns="janela",
    values="acuracia",
)

# Organiza o Macro-F1 dos classificadores por janela
comparacao_f1 = df_barras.pivot(
    index="classificador",
    columns="janela",
    values="f1_macro",
)

print("\nProcessamento concluído!")