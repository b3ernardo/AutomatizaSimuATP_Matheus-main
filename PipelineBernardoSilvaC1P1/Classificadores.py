"""
Classificação de faltas de alta impedância por região.

Otimização com Optuna: Random Forest, SVM, KNN, Regressão Logística e XGBoost.

Compatível com os caches já gerados pelo código anterior:
- hiperparametros_rf_16.json (Random Forest)
- hiperparametros_lr_16.json (Regressão Logística; renomeado para _rl_16.json)

Dependências:
    pip install numpy pandas scikit-learn optuna xgboost pyarrow matplotlib

Mantenha este arquivo ao lado de Regionalizacoes.py e DatasetAtributos16/.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from Regionalizacoes import REGIONALIZACOES


# ============================================================
# CONFIGURAÇÕES — ALTERE AQUI
# ============================================================
DIRETORIO_PIPELINE = Path(__file__).resolve().parent
DIRETORIO_DATASETS = DIRETORIO_PIPELINE / "DatasetAtributos16"
DIRETORIO_HIPERPARAMETROS = DIRETORIO_PIPELINE / "Hiperparametros"
DIRETORIO_GRAFICOS = DIRETORIO_PIPELINE / "Graficos"

JANELAS = [(0, 5), (0, 10), (0, 20)]
NUMEROS_REGIOES = list(range(2, 34))

ATRIBUTOS_DE_INTERESSE = [
    "soma_modulo_corrente_H1_maximo",
    "soma_modulo_corrente_H1_media",
    "soma_modulo_corrente_H1_minimo",
    "soma_modulo_tensao_H1_maximo",
    "soma_modulo_tensao_H1_media",
    "soma_modulo_tensao_H1_minimo",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42
N_TRIALS = 100
N_FOLDS = 5
N_ESTIMATORS = 500
N_NEIGHBORS = 5

METADADOS = ["arquivo", "fase_faltosa", "barra"]

# Quais classificadores serão executados nesta máquina?
# Como RF e RL já foram otimizados, o padrão executa apenas os três restantes.
# Para também AVALIAR RF e RL com os JSONs antigos, inclua os nomes nesta lista.
CLASSIFICADORES_ATIVOS = [
    "SVM",
    "KNN",
    "XGBoost",
    "Random Forest",
    "Regressão Logística",
]

# Controle INDIVIDUAL por classificador:
# "carregar": exige registro existente e compatível; NÃO otimiza.
# "otimizar": executa novamente, mesmo se existir registro no JSON.
# "auto": reutiliza registro compatível; caso contrário, otimiza e salva.
MODO_HIPERPARAMETROS = {
    "Random Forest": "auto",
    "SVM": "auto",
    "KNN": "auto",
    "Regressão Logística": "auto",
    "XGBoost": "auto",
}

# True: exige assinatura idêntica (inclui versões do sklearn/Optuna).
# Se RF/RL foram gerados em outra máquina com versões diferentes, pode haver
# incompatibilidade mesmo com os mesmos dados. Só use False após CONFERIR
# manualmente que datasets, partições, atributos e protocolo são idênticos.
VALIDAR_ASSINATURA_CACHE = False

# Ao escolher "otimizar", guarda cópia do JSON anterior antes de substituí-lo.
CRIAR_BACKUP_AO_REOTIMIZAR = True

NOMES_ARQUIVOS = {
    "Random Forest": "hiperparametros_rf_16.json",
    "SVM": "hiperparametros_svm_16.json",
    "KNN": "hiperparametros_knn_16.json",
    "Regressão Logística": "hiperparametros_rl_16.json",
    "XGBoost": "hiperparametros_xgboost_16.json",
}

SIGLAS = {
    "Random Forest": "RF",
    "SVM": "SVM",
    "KNN": "KNN",
    "Regressão Logística": "RL",
    "XGBoost": "XGB",
}

# Caches antigos, anteriores à pasta Hiperparametros e à padronização "rl".
# Se o arquivo de destino ainda não existir, ele será COPIADO automaticamente.
CAMINHOS_LEGADOS = {
    "Random Forest": [DIRETORIO_PIPELINE / "hiperparametros_rf_16.json"],
    "Regressão Logística": [
        DIRETORIO_HIPERPARAMETROS / "hiperparametros_lr_16.json",
        DIRETORIO_PIPELINE / "hiperparametros_lr_16.json",
        DIRETORIO_PIPELINE / "hiperparametros_rl_16.json",
    ],
}

# Documentação dos espaços de busca; também entram na assinatura dos novos caches.
ESPACOS_BUSCA = {
    "Random Forest": {
        "n_estimators": [100, 600, 100],
        "max_depth": [None, 5, 10, 20, 40],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt", "log2", None],
    },
    "Regressão Logística": {
        "C": [1e-3, 1e3, "log"],
        "class_weight": [None, "balanced"],
    },
    "SVM": {
        "C": [1e-3, 1e4, "log"],
        "gamma": [1e-4, 1e2, "log"],
        "class_weight": [None, "balanced"],
    },
    "KNN": {
        "n_neighbors": [1, 25],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "XGBoost": {
        "n_estimators": [100, 500, 100],
        "max_depth": [2, 8],
        "learning_rate": [1e-2, 3e-1, "log"],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "min_child_weight": [1, 10],
        "reg_lambda": [1e-3, 10.0, "log"],
    },
}


# ============================================================
# Dados e regionalizações (mesma metodologia do código original)
# ============================================================
def carregar_dataset(ciclos_antes, ciclos_depois):
    caminho = DIRETORIO_DATASETS / f"atributos_{ciclos_antes}_{ciclos_depois}.parquet"
    if not caminho.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {caminho}")

    df = pd.read_parquet(caminho).copy()
    colunas_ausentes = set(METADADOS + ATRIBUTOS_DE_INTERESSE) - set(df.columns)
    if colunas_ausentes:
        raise ValueError(f"Colunas ausentes em {caminho.name}: {sorted(colunas_ausentes)}")
    if df["arquivo"].duplicated().any():
        raise ValueError(f"Arquivos duplicados em {caminho.name}.")
    return df


def preparar_matriz(df):
    X = df[ATRIBUTOS_DE_INTERESSE].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Foram encontrados valores NaN ou infinitos nos atributos.")
    return X


def construir_mapa_barra_regiao(regioes):
    mapa = {}
    for nome_regiao, barras in regioes.items():
        for barra in barras:
            if barra in mapa:
                raise ValueError(f"A barra {barra} pertence a mais de uma região.")
            mapa[barra] = nome_regiao
    return mapa


def validar_regionalizacoes(barras_esperadas):
    for numero_regioes in NUMEROS_REGIOES:
        if numero_regioes not in REGIONALIZACOES:
            raise ValueError(f"Regionalização {numero_regioes} não encontrada.")
        regioes = REGIONALIZACOES[numero_regioes]
        if len(regioes) != numero_regioes:
            raise ValueError(f"Regionalização {numero_regioes}: quantidade incorreta de regiões.")
        mapa = construir_mapa_barra_regiao(regioes)
        if set(mapa) != barras_esperadas:
            raise ValueError(
                f"Regionalização {numero_regioes}: conjunto de barras incompatível com o dataset."
            )


def criar_alvo_regiao(df, numero_regioes):
    regioes = REGIONALIZACOES[numero_regioes]
    mapa = construir_mapa_barra_regiao(regioes)
    y = df["barra"].astype(int).map(mapa)
    if y.isna().any():
        raise ValueError(f"Existem barras sem região na regionalização {numero_regioes}.")
    return y


# ============================================================
# Construção dos cinco classificadores
# ============================================================
def criar_classificador(nome, parametros=None):
    parametros = dict(parametros or {})

    if nome == "Random Forest":
        padrao = {"n_estimators": N_ESTIMATORS, "random_state": RANDOM_STATE, "n_jobs": -1}
        padrao.update(parametros)
        return RandomForestClassifier(**padrao)

    if nome == "SVM":
        padrao = {"kernel": "rbf"}
        padrao.update(parametros)
        return make_pipeline(StandardScaler(), SVC(**padrao))

    if nome == "KNN":
        padrao = {"n_neighbors": N_NEIGHBORS, "n_jobs": -1}
        padrao.update(parametros)
        return make_pipeline(StandardScaler(), KNeighborsClassifier(**padrao))

    if nome == "Regressão Logística":
        padrao = {"solver": "lbfgs", "max_iter": 2000, "random_state": RANDOM_STATE}
        padrao.update(parametros)
        return make_pipeline(StandardScaler(), LogisticRegression(**padrao))

    if nome == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "Para executar XGBoost, instale a dependência: pip install xgboost"
            ) from exc
        padrao = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "verbosity": 0,
        }
        padrao.update(parametros)
        return XGBClassifier(**padrao)

    raise ValueError(f"Classificador desconhecido: {nome}")


# ============================================================
# Validação cruzada e Optuna
# ============================================================
def criar_validacao():
    return StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )


def avaliar_validacao(modelo, X_treino, y_treino):
    pontuacoes = cross_val_score(
        modelo, X_treino, y_treino,
        cv=criar_validacao(), scoring="f1_macro", n_jobs=1,
        error_score="raise",
    )
    return float(np.mean(pontuacoes))


def executar_optuna(objetivo):
    estudo = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    estudo.optimize(objetivo, n_trials=N_TRIALS)
    return estudo.best_params, float(estudo.best_value)


def otimizar_random_forest(X_treino, y_treino):
    # Mesmo espaço de busca do script original: mantém compatibilidade do cache.
    def objetivo(trial):
        parametros = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=100),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 10, 20, 40]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
        return avaliar_validacao(criar_classificador("Random Forest", parametros), X_treino, y_treino)

    return executar_optuna(objetivo)


def otimizar_svm(X_treino, y_treino):
    def objetivo(trial):
        parametros = {
            "C": trial.suggest_float("C", 1e-3, 1e4, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e3, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        return avaliar_validacao(criar_classificador("SVM", parametros), X_treino, y_treino)

    return executar_optuna(objetivo)


def otimizar_knn(X_treino, y_treino):
    def objetivo(trial):
        parametros = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 25),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_categorical("p", [1, 2]),
        }
        return avaliar_validacao(criar_classificador("KNN", parametros), X_treino, y_treino)

    return executar_optuna(objetivo)


def otimizar_regressao_logistica(X_treino, y_treino):
    # Mesmo espaço de busca e pipeline da otimização RL já executada.
    def objetivo(trial):
        parametros = {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        return avaliar_validacao(
            criar_classificador("Regressão Logística", parametros), X_treino, y_treino
        )

    return executar_optuna(objetivo)


def otimizar_xgboost(X_treino, y_treino):
    # XGBoost requer classes 0, 1, ..., k-1. O LabelEncoder é ajustado
    # apenas com os rótulos de treinamento (não com o conjunto de teste).
    codificador = LabelEncoder()
    y_codificado = codificador.fit_transform(y_treino)

    def objetivo(trial):
        parametros = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        return avaliar_validacao(
            criar_classificador("XGBoost", parametros), X_treino, y_codificado
        )

    return executar_optuna(objetivo)


FUNCOES_OTIMIZACAO = {
    "Random Forest": otimizar_random_forest,
    "SVM": otimizar_svm,
    "KNN": otimizar_knn,
    "Regressão Logística": otimizar_regressao_logistica,
    "XGBoost": otimizar_xgboost,
}


# ============================================================
# Assinaturas: preservam compatibilidade com RF e RL antigos
# ============================================================
def assinatura_de_resumo(resumo, X_treino, y_treino):
    assinatura = hashlib.sha256(json.dumps(resumo, sort_keys=True).encode("utf-8"))
    assinatura.update(pd.util.hash_pandas_object(X_treino, index=True).values.tobytes())
    assinatura.update(pd.util.hash_pandas_object(y_treino, index=True).values.tobytes())
    return assinatura.hexdigest()


def assinatura_otimizacao_rf(X_treino, y_treino):
    # NÃO alterar sem necessidade: idêntica à assinatura do script RF original.
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
    return assinatura_de_resumo(resumo, X_treino, y_treino)


def assinatura_otimizacao_rl(X_treino, y_treino):
    # NÃO alterar sem necessidade: idêntica à assinatura do script RL anterior.
    resumo = {
        "atributos": ATRIBUTOS_DE_INTERESSE,
        "n_trials": N_TRIALS,
        "n_folds": N_FOLDS,
        "random_state": RANDOM_STATE,
        "sklearn": sklearn.__version__,
        "optuna": optuna.__version__,
        "preprocessamento": "StandardScaler no pipeline",
        "solver": "lbfgs",
        "regularizacao": "L2 (padrao)",
        "max_iter": 2000,
        "busca": {
            "C": [1e-3, 1e3, "log"],
            "class_weight": [None, "balanced"],
        },
    }
    return assinatura_de_resumo(resumo, X_treino, y_treino)


def assinatura_otimizacao(nome, X_treino, y_treino):
    if nome == "Random Forest":
        return assinatura_otimizacao_rf(X_treino, y_treino)
    if nome == "Regressão Logística":
        return assinatura_otimizacao_rl(X_treino, y_treino)

    resumo = {
        "atributos": ATRIBUTOS_DE_INTERESSE,
        "n_trials": N_TRIALS,
        "n_folds": N_FOLDS,
        "random_state": RANDOM_STATE,
        "sklearn": sklearn.__version__,
        "optuna": optuna.__version__,
        "busca": ESPACOS_BUSCA[nome],
        "configuracao_fixa": {
            "SVM": {"scaler": "StandardScaler", "kernel": "rbf"},
            "KNN": {"scaler": "StandardScaler", "metric": "minkowski"},
            "XGBoost": {"tree_method": "hist", "eval_metric": "mlogloss"},
        }[nome],
    }
    if nome == "XGBoost":
        import xgboost
        resumo["xgboost"] = xgboost.__version__
    return assinatura_de_resumo(resumo, X_treino, y_treino)


# ============================================================
# Arquivos JSON e política de carregar / otimizar / auto
# ============================================================
def caminho_cache(nome):
    return DIRETORIO_HIPERPARAMETROS / NOMES_ARQUIVOS[nome]


def carregar_cache(nome):
    DIRETORIO_HIPERPARAMETROS.mkdir(parents=True, exist_ok=True)
    destino = caminho_cache(nome)

    if not destino.is_file():
        for origem in CAMINHOS_LEGADOS.get(nome, []):
            if origem.is_file():
                shutil.copy2(origem, destino)
                print(f"[{SIGLAS[nome]}] Cache anterior copiado: {origem.name} -> {destino}", flush=True)
                break

    if not destino.is_file():
        return {}

    with destino.open("r", encoding="utf-8") as arquivo:
        cache = json.load(arquivo)
    if not isinstance(cache, dict):
        raise ValueError(f"Formato de cache inválido: {destino}")
    return cache


def salvar_cache(nome, cache):
    destino = caminho_cache(nome)
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".json.tmp")
    try:
        with temporario.open("w", encoding="utf-8") as arquivo:
            json.dump(cache, arquivo, ensure_ascii=False, indent=4)
        os.replace(temporario, destino)
    finally:
        if temporario.exists():
            temporario.unlink()


def criar_backup_cache(nome):
    destino = caminho_cache(nome)
    if not destino.is_file():
        return
    horario = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = destino.with_name(f"{destino.stem}.backup_{horario}.json")
    shutil.copy2(destino, backup)
    print(f"[{SIGLAS[nome]}] Backup criado: {backup.name}", flush=True)


def obter_hiperparametros(nome, X_treino, y_treino, nome_janela, numero_regioes, cache):
    modo = MODO_HIPERPARAMETROS[nome]
    chave = f"{nome_janela}|{numero_regioes}"
    assinatura = assinatura_otimizacao(nome, X_treino, y_treino)
    registro = cache.get(chave)

    possui_registro = (
        isinstance(registro, dict)
        and isinstance(registro.get("parametros"), dict)
        and "f1_validacao" in registro
    )
    assinatura_igual = possui_registro and registro.get("assinatura") == assinatura
    reutilizavel = possui_registro and (
        assinatura_igual or not VALIDAR_ASSINATURA_CACHE
    )

    if modo != "otimizar" and reutilizavel:
        if not assinatura_igual:
            print(
                f"[AVISO] [{SIGLAS[nome]}] {chave}: assinatura divergente. "
                "Usando cache SOMENTE porque VALIDAR_ASSINATURA_CACHE=False. "
                "Confirme que os dados e o protocolo experimental são os mesmos!",
                flush=True,
            )
        print(
            f"[{SIGLAS[nome]}] Janela {nome_janela} | {numero_regioes} regiões | "
            "hiperparâmetros recuperados do JSON",
            flush=True,
        )
        return registro["parametros"], registro["f1_validacao"]

    if modo == "carregar":
        motivo = (
            "registro não encontrado" if not possui_registro
            else "assinatura incompatível com os dados, configurações ou versões"
        )
        raise ValueError(
            f"[{SIGLAS[nome]}] Não foi possível carregar {chave}: {motivo}. "
            f"Arquivo: {caminho_cache(nome)}. "
            "Use modo 'auto'/'otimizar' para calcular novamente. "
            "Se o cache veio de outra máquina e você verificou a equivalência "
            "dos dados e do experimento, pode desativar explicitamente "
            "VALIDAR_ASSINATURA_CACHE."
        )

    contagens = pd.Series(y_treino).value_counts()
    if len(contagens) < 2 or contagens.min() < N_FOLDS:
        raise ValueError(
            f"[{SIGLAS[nome]}] {chave}: a menor classe de treinamento tem "
            f"{int(contagens.min())} amostras; são necessárias pelo menos "
            f"{N_FOLDS} por classe para StratifiedKFold({N_FOLDS})."
        )

    print(
        f"[{SIGLAS[nome]}] Otimizando janela {nome_janela} | "
        f"{numero_regioes} regiões ({N_TRIALS} tentativas)...",
        flush=True,
    )
    melhores_parametros, f1_validacao = FUNCOES_OTIMIZACAO[nome](X_treino, y_treino)

    cache[chave] = {
        "parametros": melhores_parametros,
        "f1_validacao": f1_validacao,
        "assinatura": assinatura,
    }
    salvar_cache(nome, cache)  # Salva a cada regionalização, mesmo se houver interrupção.
    print(
        f"[{SIGLAS[nome]}] Janela {nome_janela} | {numero_regioes} regiões | "
        f"Macro-F1 validação: {f1_validacao:.4f} | hiperparâmetros salvos",
        flush=True,
    )
    return melhores_parametros, f1_validacao


# ============================================================
# Gráficos de evolução: desempenho no conjunto de teste
# ============================================================
def gerar_grafico_evolucao(df_resultados):
    """Gera um PDF vetorial com 6 páginas: uma métrica e uma janela por página."""
    DIRETORIO_GRAFICOS.mkdir(parents=True, exist_ok=True)
    caminho_pdf = DIRETORIO_GRAFICOS / "evolucao_classificadores.pdf"

    metricas = [("acuracia", "Acurácia"), ("f1_macro", "Macro-F1")]
    marcacoes_x = [n for n in [2, 5, 10, 15, 20, 25, 30, 33] if n in NUMEROS_REGIOES]
    if not marcacoes_x:
        marcacoes_x = sorted(NUMEROS_REGIOES)
    cores = plt.get_cmap("tab10")

    with PdfPages(caminho_pdf) as pdf:
        for ciclos_antes, ciclos_depois in JANELAS:
            nome_janela = f"{ciclos_antes}/{ciclos_depois}"
            df_janela = df_resultados[df_resultados["janela"] == nome_janela]

            for coluna, titulo in metricas:
                # Cada figura ocupa uma página inteira, em orientação paisagem.
                fig, eixo = plt.subplots(figsize=(15.5, 9))
                linhas_legenda = []

                for indice_classificador, nome in enumerate(CLASSIFICADORES_ATIVOS):
                    dados = (
                        df_janela[df_janela["classificador"] == nome]
                        .sort_values("n_regioes")
                    )
                    if dados.empty:
                        continue
                    linha, = eixo.plot(
                        dados["n_regioes"], dados[coluna],
                        color=cores(indice_classificador % 10),
                        linewidth=2.2, marker="o", markersize=4,
                        label=nome,
                    )
                    linhas_legenda.append(linha)

                eixo.set_title(
                    f"{titulo} por número de regiões — janela {nome_janela}",
                    fontsize=19, fontweight="bold", pad=20,
                )
                eixo.set_xlabel("Número de regiões", fontsize=14, labelpad=12)
                eixo.set_ylabel(titulo, fontsize=14, labelpad=12)
                eixo.set_xlim(min(NUMEROS_REGIOES) - 0.5, max(NUMEROS_REGIOES) + 0.5)
                eixo.set_ylim(0, 1.02)
                eixo.set_xticks(marcacoes_x)
                eixo.set_yticks(np.arange(0, 1.01, 0.1))
                eixo.tick_params(axis="both", labelsize=12)
                eixo.grid(True, linestyle="--", alpha=0.3)
                eixo.spines["top"].set_visible(False)
                eixo.spines["right"].set_visible(False)

                fig.text(
                    0.5, 0.93, "Desempenho no conjunto de teste",
                    ha="center", fontsize=12, color="dimgray",
                )
                if linhas_legenda:
                    fig.legend(
                        handles=linhas_legenda,
                        labels=[linha.get_label() for linha in linhas_legenda],
                        loc="lower center", ncol=min(len(linhas_legenda), 5),
                        bbox_to_anchor=(0.5, 0.065),
                        frameon=False, fontsize=12,
                    )
                fig.subplots_adjust(left=0.08, right=0.975, top=0.86, bottom=0.19)
                pdf.savefig(fig)
                plt.close(fig)

    print(f"\nPDF com {len(JANELAS) * len(metricas)} gráficos salvo em: {caminho_pdf}")

# ============================================================
# Execução
# ============================================================
def validar_configuracoes():
    if not JANELAS:
        raise ValueError("A lista JANELAS está vazia.")
    if not NUMEROS_REGIOES:
        raise ValueError("A lista NUMEROS_REGIOES está vazia.")
    if not CLASSIFICADORES_ATIVOS:
        raise ValueError("Selecione pelo menos um classificador em CLASSIFICADORES_ATIVOS.")
    if len(CLASSIFICADORES_ATIVOS) != len(set(CLASSIFICADORES_ATIVOS)):
        raise ValueError("CLASSIFICADORES_ATIVOS contém nomes duplicados.")
    for nome in CLASSIFICADORES_ATIVOS:
        if nome not in FUNCOES_OTIMIZACAO:
            raise ValueError(f"Classificador inválido em CLASSIFICADORES_ATIVOS: {nome}")
        if MODO_HIPERPARAMETROS.get(nome) not in {"carregar", "otimizar", "auto"}:
            raise ValueError(f"Modo inválido para {nome}: {MODO_HIPERPARAMETROS.get(nome)}")
    if N_FOLDS < 2 or N_TRIALS < 1:
        raise ValueError("N_FOLDS deve ser >= 2 e N_TRIALS >= 1.")
    if "XGBoost" in CLASSIFICADORES_ATIVOS:
        criar_classificador("XGBoost")  # Falha cedo com mensagem de instalação.


def executar():
    validar_configuracoes()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ciclos_antes_base, ciclos_depois_base = JANELAS[0]
    df_base = carregar_dataset(ciclos_antes_base, ciclos_depois_base)
    barras_esperadas = set(df_base["barra"].astype(int))
    validar_regionalizacoes(barras_esperadas)

    # Exatamente o mesmo split do código original: estratificação pela barra.
    arquivos_treino, arquivos_teste = train_test_split(
        df_base["arquivo"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_base["barra"],
    )
    arquivos_treino = set(arquivos_treino)
    arquivos_teste = set(arquivos_teste)
    arquivos_esperados = arquivos_treino.union(arquivos_teste)
    if arquivos_treino.intersection(arquivos_teste):
        raise RuntimeError("Há simulações compartilhadas entre treino e teste.")

    # As classes por arquivo devem permanecer as mesmas nas três janelas.
    barra_por_arquivo_base = df_base.set_index("arquivo")["barra"].astype(int).sort_index()

    print("=" * 100)
    print("CLASSIFICAÇÃO DE FALTAS DE ALTA IMPEDÂNCIA")
    print("=" * 100)
    print(f"Treinamento: {len(arquivos_treino)} simulações")
    print(f"Teste: {len(arquivos_teste)} simulações")
    print(f"Atributos: {len(ATRIBUTOS_DE_INTERESSE)}")
    print(f"Regionalizações: {len(NUMEROS_REGIOES)}")
    print(f"Classificadores ativos: {', '.join(CLASSIFICADORES_ATIVOS)}")
    print(f"Pasta dos hiperparâmetros: {DIRETORIO_HIPERPARAMETROS}")

    caches = {}
    for nome in CLASSIFICADORES_ATIVOS:
        caches[nome] = carregar_cache(nome)
        modo = MODO_HIPERPARAMETROS[nome]
        print(f"[{SIGLAS[nome]}] Modo: {modo} | Arquivo: {caminho_cache(nome).name}")
        if modo == "otimizar" and CRIAR_BACKUP_AO_REOTIMIZAR:
            criar_backup_cache(nome)

    resultados = []
    for ciclos_antes, ciclos_depois in JANELAS:
        nome_janela = f"{ciclos_antes}/{ciclos_depois}"
        print("\n" + "=" * 100)
        print(f"JANELA {nome_janela}")
        print("=" * 100)

        df = carregar_dataset(ciclos_antes, ciclos_depois)
        if set(df["arquivo"]) != arquivos_esperados:
            raise ValueError(f"A janela {nome_janela} não contém as mesmas simulações.")
        if set(df["barra"].astype(int)) != barras_esperadas:
            raise ValueError(f"A janela {nome_janela} contém barras incompatíveis.")
        if not df.set_index("arquivo")["barra"].astype(int).sort_index().equals(barra_por_arquivo_base):
            raise ValueError(f"A janela {nome_janela} associa barras diferentes aos arquivos.")

        X = preparar_matriz(df)
        mascara_treino = df["arquivo"].isin(arquivos_treino)
        mascara_teste = df["arquivo"].isin(arquivos_teste)
        X_treino, X_teste = X.loc[mascara_treino], X.loc[mascara_teste]

        for numero_regioes in NUMEROS_REGIOES:
            y = criar_alvo_regiao(df, numero_regioes)
            y_treino, y_teste = y.loc[mascara_treino], y.loc[mascara_teste]
            if y_treino.nunique() != numero_regioes:
                raise ValueError(
                    f"Regionalização {numero_regioes}: existem regiões ausentes no treinamento."
                )

            for nome in CLASSIFICADORES_ATIVOS:
                melhores_parametros, f1_validacao = obter_hiperparametros(
                    nome, X_treino, y_treino, nome_janela, numero_regioes, caches[nome]
                )
                modelo = criar_classificador(nome, melhores_parametros)

                if nome == "XGBoost":
                    # Os rótulos podem ser strings; XGBoost precisa de 0..k-1.
                    codificador = LabelEncoder()
                    y_treino_codificado = codificador.fit_transform(y_treino)
                    modelo.fit(X_treino, y_treino_codificado)
                    y_predito = codificador.inverse_transform(
                        np.asarray(modelo.predict(X_teste), dtype=int)
                    )
                else:
                    modelo.fit(X_treino, y_treino)
                    y_predito = modelo.predict(X_teste)

                acuracia = accuracy_score(y_teste, y_predito)
                f1_macro = f1_score(y_teste, y_predito, average="macro", zero_division=0)
                resultados.append({
                    "janela": nome_janela,
                    "n_regioes": numero_regioes,
                    "classificador": nome,
                    "acuracia": acuracia,
                    "f1_macro": f1_macro,
                    "f1_validacao": f1_validacao,
                })
                print(
                    f"[{SIGLAS[nome]}] Teste | {nome_janela} | {numero_regioes} regiões | "
                    f"Acurácia: {acuracia:.4f} | Macro-F1: {f1_macro:.4f}",
                    flush=True,
                )

        df_janela = pd.DataFrame(resultados)
        df_janela = df_janela[df_janela["janela"] == nome_janela]
        for coluna, titulo in [
            ("acuracia", "ACURÁCIA POR NÚMERO DE REGIÕES"),
            ("f1_macro", "MACRO-F1 POR NÚMERO DE REGIÕES"),
        ]:
            tabela = df_janela.pivot(index="n_regioes", columns="classificador", values=coluna)
            print(f"\n{titulo}")
            print(tabela.to_string(float_format=lambda valor: f"{valor:.4f}"))

    df_resultados = pd.DataFrame(resultados)
    numero_maximo_regioes = max(NUMEROS_REGIOES)
    df_maximo = df_resultados[df_resultados["n_regioes"] == numero_maximo_regioes]
    for coluna, titulo in [
        ("acuracia", "COMPARAÇÃO FINAL — ACURÁCIA"),
        ("f1_macro", "COMPARAÇÃO FINAL — MACRO-F1"),
    ]:
        tabela = df_maximo.pivot(index="classificador", columns="janela", values=coluna)
        print(f"\n{titulo} ({numero_maximo_regioes} regiões)")
        print(tabela.to_string(float_format=lambda valor: f"{valor:.4f}"))

    gerar_grafico_evolucao(df_resultados)

    print("\nProcessamento concluído! Hiperparâmetros salvos nos arquivos JSON.")


if __name__ == "__main__":
    executar()
