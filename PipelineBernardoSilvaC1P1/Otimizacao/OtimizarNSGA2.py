import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

# Permite importar as regionalizações da pasta principal da pipeline
DIRETORIO_PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DIRETORIO_PIPELINE))
from Regionalizacoes import REGIONALIZACOES


# ============================================================
# Configurações
# ============================================================
DIRETORIO_OTIMIZACAO = Path(__file__).resolve().parent

CAMINHO_DATASET_ORIGINAL = (
    DIRETORIO_PIPELINE / "DatasetAtributos16" / "atributos_0_5.parquet"
)

CAMINHO_DATASET_MEDIDORES = (
    DIRETORIO_OTIMIZACAO / "DatasetAtributosMedidores16" / "atributos_0_5.parquet"
)

CAMINHO_CACHE_RF = DIRETORIO_PIPELINE / "hiperparametros_rf_16.json"
DIRETORIO_RESULTADOS = DIRETORIO_OTIMIZACAO / "ResultadosNSGA2"

NUMERO_REGIOES = 8
NOME_JANELA = "0/5"
TEST_SIZE = 0.20
RANDOM_STATE = 42
N_ESTIMATORS = 500
N_FOLDS = 5

MAXIMO_MEDIDORES = 7
DESEMPENHO_MINIMO = 0.90
TAMANHO_POPULACAO = 100
NUMERO_GERACOES = 50
SEMENTE = 42

ATRIBUTOS_DE_INTERESSE = [
    "soma_modulo_corrente_H1_maximo",
    "soma_modulo_corrente_H1_media",
    "soma_modulo_corrente_H1_minimo",
    "soma_modulo_tensao_H1_maximo",
    "soma_modulo_tensao_H1_media",
    "soma_modulo_tensao_H1_minimo",
]

# Uma posição binária para cada barra do IEEE 34.
MEDIDORES_TODOS = [
    "SUB",
    "M02", "M03", "M04", "M05", "M06", "M07", "M08",
    "M09", "M10", "M11", "M12", "M13", "M14", "M15",
    "M16", "M17", "M18", "M19", "M20", "M21", "M22",
    "M23", "M24", "M25", "M26", "M27", "M28", "M29",
    "M30", "M31", "M32", "M33", "M34",
]

# Medidores não disponíveis no sistema.
# As posições correspondentes permanecem sempre iguais a zero.
MEDIDORES_INDISPONIVEIS = {
    "M09", "M10", "M12", "M13", "M15",
    "M18", "M23", "M30", "M31",
}

# ============================================================
# Preparação dos dados e dos hiperparâmetros
# ============================================================
def carregar_hiperparametros():
    if not CAMINHO_CACHE_RF.is_file():
        raise FileNotFoundError(
            f"JSON de hiperparâmetros não encontrado: {CAMINHO_CACHE_RF}"
        )

    with CAMINHO_CACHE_RF.open("r", encoding="utf-8") as arquivo:
        cache = json.load(arquivo)

    chave = f"{NOME_JANELA}|{NUMERO_REGIOES}"
    registro = cache.get(chave)

    if not isinstance(registro, dict) or not isinstance(
        registro.get("parametros"), dict
    ):
        raise ValueError(
            f"Hiperparâmetros não encontrados no JSON para a chave '{chave}'."
        )

    # Reutiliza os hiperparâmetros otimizados originalmente para a SUB.
    # A assinatura antiga não deve ser comparada, pois as colunas mudaram.
    return registro["parametros"]


# ============================================================
# Avaliação das combinações de medidores
# ============================================================
class AvaliadorMedidores:
    def __init__(self):
        if not CAMINHO_DATASET_ORIGINAL.is_file():
            raise FileNotFoundError(
                f"Dataset original não encontrado: {CAMINHO_DATASET_ORIGINAL}"
            )

        if not CAMINHO_DATASET_MEDIDORES.is_file():
            raise FileNotFoundError(
                f"Dataset dos medidores não encontrado: {CAMINHO_DATASET_MEDIDORES}"
            )

        df_original = pd.read_parquet(CAMINHO_DATASET_ORIGINAL)
        df_medidores = pd.read_parquet(CAMINHO_DATASET_MEDIDORES)

        for nome, df in [
            ("original", df_original),
            ("medidores", df_medidores),
        ]:
            if df["arquivo"].duplicated().any():
                raise ValueError(
                    f"Há simulações duplicadas no dataset {nome}."
                )

        if set(df_original["arquivo"]) != set(df_medidores["arquivo"]):
            raise ValueError(
                "Os datasets original e dos medidores não contêm "
                "as mesmas simulações."
            )

        # Reproduz o split do ClassificadorFinal.py:
        # mesma ordem do dataset original, estratificação por barra
        # e mesma semente.
        arquivos_treino, arquivos_teste = train_test_split(
            df_original["arquivo"],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=df_original["barra"],
        )

        self.arquivos_treino = list(arquivos_treino)
        self.arquivos_teste = list(arquivos_teste)

        self.df_treino = (
            df_medidores.set_index("arquivo").loc[self.arquivos_treino]
        )

        df_original_treino = (
            df_original.set_index("arquivo").loc[self.arquivos_treino]
        )

        if not np.array_equal(
            self.df_treino["barra"].to_numpy(dtype=int),
            df_original_treino["barra"].to_numpy(dtype=int),
        ):
            raise ValueError(
                "As barras não coincidem entre os dois datasets."
            )

        # Mantém a regionalização do classificador original.
        mapa_barra_regiao = {}
        regioes = REGIONALIZACOES[NUMERO_REGIOES]

        if len(regioes) != NUMERO_REGIOES:
            raise ValueError(
                "Número incorreto de regiões na regionalização."
            )

        for regiao, barras in regioes.items():
            for barra in barras:
                if barra in mapa_barra_regiao:
                    raise ValueError(
                        f"Barra {barra} presente em mais de uma região."
                    )

                mapa_barra_regiao[barra] = regiao

        self.barras_treino = self.df_treino["barra"].astype(int)
        self.y_treino = self.barras_treino.map(mapa_barra_regiao)

        if (
            self.y_treino.isna().any()
            or self.y_treino.nunique() != NUMERO_REGIOES
        ):
            raise ValueError(
                "Regionalização incompatível com as barras do treinamento."
            )

        # Os mesmos folds são reutilizados para todas as combinações.
        validacao = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        self.folds = list(
            validacao.split(self.df_treino, self.barras_treino)
        )

        self.hiperparametros = carregar_hiperparametros()
        self.regioes = sorted(regioes)
        self.cache_avaliacoes = {}

        # Identifica automaticamente os medidores disponíveis.
        self.medidores_disponiveis = sorted({
            coluna.split("_", 1)[0]
            for coluna in df_medidores.columns
            if any(
                coluna.endswith("_" + atributo)
                for atributo in ATRIBUTOS_DE_INTERESSE
            )
        })

        # Coloca a SUB na primeira posição.
        if "SUB" in self.medidores_disponiveis:
            self.medidores_disponiveis.remove("SUB")
            self.medidores_disponiveis.insert(0, "SUB")

        # Verifica se os medidores encontrados no dataset correspondem
        # exatamente às posições consideradas disponíveis.
        medidores_esperados = [
            medidor
            for medidor in MEDIDORES_TODOS
            if medidor not in MEDIDORES_INDISPONIVEIS
        ]

        if set(self.medidores_disponiveis) != set(medidores_esperados):
            faltantes = (
                set(medidores_esperados)
                - set(self.medidores_disponiveis)
            )

            extras = (
                set(self.medidores_disponiveis)
                - set(medidores_esperados)
            )

            raise ValueError(
                "Os medidores disponíveis no dataset não coincidem "
                f"com a configuração esperada. Faltantes: {sorted(faltantes)}. "
                f"Extras: {sorted(extras)}."
            )

        # Verifica se cada medidor possui os seis atributos.
        for medidor in self.medidores_disponiveis:
            colunas = [
                f"{medidor}_{atributo}"
                for atributo in ATRIBUTOS_DE_INTERESSE
            ]

            faltantes = set(colunas) - set(df_medidores.columns)

            if faltantes:
                raise ValueError(
                    f"Atributos ausentes para {medidor}: "
                    f"{sorted(faltantes)}"
                )

        print(f"Treinamento: {len(self.arquivos_treino)} simulações")
        print(f"Teste reservado: {len(self.arquivos_teste)} simulações")
        print(
            f"Representação binária: {len(MEDIDORES_TODOS)} posições"
        )
        print(
            f"Medidores disponíveis ({len(self.medidores_disponiveis)}): "
            f"{self.medidores_disponiveis}"
        )
        print(
            f"Medidores indisponíveis ({len(MEDIDORES_INDISPONIVEIS)}): "
            f"{sorted(MEDIDORES_INDISPONIVEIS)}"
        )
        print(
            f"Validação cruzada: {N_FOLDS} folds, "
            "estratificados por barra"
        )
        print(
            f"Hiperparâmetros: JSON original, "
            f"chave '{NOME_JANELA}|{NUMERO_REGIOES}'"
        )

    def avaliar(self, medidores):
        if not medidores:
            raise ValueError(
                "É necessário selecionar pelo menos um medidor."
            )

        if len(medidores) != len(set(medidores)):
            raise ValueError(
                "Há medidores repetidos na combinação."
            )

        desconhecidos = (
            set(medidores) - set(self.medidores_disponiveis)
        )

        if desconhecidos:
            raise ValueError(
                f"Medidores desconhecidos: {sorted(desconhecidos)}"
            )

        # Ordenação canônica para reutilizar avaliações.
        chave = tuple(
            medidor
            for medidor in self.medidores_disponiveis
            if medidor in medidores
        )

        if chave in self.cache_avaliacoes:
            return self.cache_avaliacoes[chave]

        colunas = [
            f"{medidor}_{atributo}"
            for medidor in chave
            for atributo in ATRIBUTOS_DE_INTERESSE
        ]

        X = self.df_treino[colunas].to_numpy(dtype=float)
        y = self.y_treino.to_numpy()

        if not np.isfinite(X).all():
            raise ValueError(
                f"NaN ou infinito nos atributos da combinação {chave}."
            )

        resultados_folds = []

        for indices_treino, indices_validacao in self.folds:
            modelo = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            modelo.set_params(**self.hiperparametros)

            # Garante consistência entre as avaliações.
            modelo.set_params(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

            modelo.fit(
                X[indices_treino],
                y[indices_treino],
            )

            y_predito = modelo.predict(
                X[indices_validacao]
            )

            f1 = f1_score(
                y[indices_validacao],
                y_predito,
                average="macro",
                labels=self.regioes,
                zero_division=0,
            )

            resultados_folds.append(float(f1))

        resultado = {
            "medidores": list(chave),
            "n_medidores": len(chave),
            "f1_macro_medio": float(np.mean(resultados_folds)),
            "f1_macro_desvio": float(np.std(resultados_folds)),
            "f1_por_fold": resultados_folds,
        }

        self.cache_avaliacoes[chave] = resultado

        return resultado


# ============================================================
# Cache persistente das avaliações do Random Forest
# ============================================================
def calcular_assinatura():
    """Impede reutilizar resultados quando dados ou configurações mudam."""

    hash_objeto = hashlib.sha256()

    configuracao = (
        f"rf={N_ESTIMATORS}|folds={N_FOLDS}|seed={RANDOM_STATE}|"
        f"regioes={NUMERO_REGIOES}|maximo={MAXIMO_MEDIDORES}|"
        "objetivo=macro_f1_medio|versao=1"
    )

    hash_objeto.update(configuracao.encode("utf-8"))

    for caminho in [
        CAMINHO_DATASET_ORIGINAL,
        CAMINHO_DATASET_MEDIDORES,
        CAMINHO_CACHE_RF,
    ]:
        with caminho.open("rb") as arquivo:
            for bloco in iter(
                lambda: arquivo.read(1024 * 1024),
                b"",
            ):
                hash_objeto.update(bloco)

    return hash_objeto.hexdigest()


class CacheAvaliacoes:
    def __init__(self, avaliador):
        self.avaliador = avaliador
        self.caminho = (
            DIRETORIO_RESULTADOS / "avaliacoes.jsonl"
        )

        self.assinatura = calcular_assinatura()
        self.novas_avaliacoes = 0

        recuperadas = 0

        if self.caminho.is_file():
            with self.caminho.open(
                "r", encoding="utf-8"
            ) as arquivo:

                for linha in arquivo:
                    if not linha.strip():
                        continue

                    registro = json.loads(linha)

                    if registro.get("assinatura") != self.assinatura:
                        continue

                    resultado = registro["resultado"]
                    chave = tuple(resultado["medidores"])

                    self.avaliador.cache_avaliacoes[chave] = resultado
                    recuperadas += 1

        print(
            f"Avaliações recuperadas do cache: {recuperadas}",
            flush=True,
        )

    def avaliar(self, medidores):
        chave = tuple(
            medidor
            for medidor in self.avaliador.medidores_disponiveis
            if medidor in medidores
        )

        if chave in self.avaliador.cache_avaliacoes:
            return self.avaliador.cache_avaliacoes[chave]

        inicio = time.perf_counter()

        resultado = self.avaliador.avaliar(list(chave))

        registro = {
            "assinatura": self.assinatura,
            "resultado": resultado,
        }

        with self.caminho.open(
            "a", encoding="utf-8"
        ) as arquivo:
            arquivo.write(
                json.dumps(registro, ensure_ascii=False) + "\n"
            )

        self.novas_avaliacoes += 1

        duracao = time.perf_counter() - inicio

        print(
            f"Avaliação {self.novas_avaliacoes}: "
            f"N={len(chave)} | "
            f"Macro-F1={resultado['f1_macro_medio']:.4f} | "
            f"{duracao:.1f}s | {list(chave)}",
            flush=True,
        )

        return resultado


# ============================================================
# Representação binária e operadores de seleção
# ============================================================
class AmostragemValida(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros(
            (n_samples, problem.n_var),
            dtype=bool,
        )

        for i in range(n_samples):
            quantidade = np.random.randint(
                1, MAXIMO_MEDIDORES + 1
            )

            escolhidos = np.random.choice(
                problem.indices_disponiveis,
                quantidade,
                replace=False,
            )

            X[i, escolhidos] = True

        return X


class RepararSelecao(Repair):
    def _do(self, problem, X, **kwargs):
        X = np.asarray(X, dtype=bool).copy()

        for individuo in X:
            # Os pontos sem medidor permanecem sempre iguais a zero.
            individuo[problem.indices_indisponiveis] = False

            selecionados = np.flatnonzero(individuo)

            if len(selecionados) > MAXIMO_MEDIDORES:
                remover = np.random.choice(
                    selecionados,
                    len(selecionados) - MAXIMO_MEDIDORES,
                    replace=False,
                )

                individuo[remover] = False

            if not individuo.any():
                individuo[
                    np.random.choice(problem.indices_disponiveis)
                ] = True

        return X


# ============================================================
# Problema multiobjetivo
# Minimizar número de medidores e (1 - Macro-F1)
# ============================================================
class ProblemaMedidores(ElementwiseProblem):
    def __init__(self, cache):
        self.cache = cache
        self.medidores = MEDIDORES_TODOS

        self.indices_disponiveis = np.array([
            i
            for i, medidor in enumerate(self.medidores)
            if medidor not in MEDIDORES_INDISPONIVEIS
        ])

        self.indices_indisponiveis = np.array([
            i
            for i, medidor in enumerate(self.medidores)
            if medidor in MEDIDORES_INDISPONIVEIS
        ])

        super().__init__(
            n_var=len(self.medidores),
            n_obj=2,
            n_ieq_constr=1,
            xl=0,
            xu=1,
            vtype=bool,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        selecionados = [
            medidor
            for medidor, ativo in zip(self.medidores, x)
            if ativo
        ]

        if not 1 <= len(selecionados) <= MAXIMO_MEDIDORES:
            raise ValueError(
                f"Indivíduo inválido: {selecionados}"
            )

        indisponiveis_selecionados = (
            set(selecionados) & MEDIDORES_INDISPONIVEIS
        )

        if indisponiveis_selecionados:
            raise ValueError(
                "Medidores indisponíveis selecionados: "
                f"{sorted(indisponiveis_selecionados)}"
            )

        resultado = self.cache.avaliar(selecionados)

        macro_f1 = resultado["f1_macro_medio"]

        out["F"] = [
            len(selecionados),
            1.0 - macro_f1,
        ]

        # No pymoo, uma restrição de desigualdade é satisfeita
        # quando G <= 0.
        out["G"] = [
            DESEMPENHO_MINIMO - macro_f1,
        ]


# ============================================================
# Exportação das soluções de Pareto
# ============================================================
def salvar_pareto(resultado, problema):
    if resultado.X is None:
        raise RuntimeError(
            "Nenhuma solução viável foi encontrada."
        )

    linhas = []

    for individuo in np.atleast_2d(resultado.X):
        medidores = [
            medidor
            for medidor, ativo in zip(
                problema.medidores,
                individuo,
            )
            if ativo
        ]

        avaliacao = problema.cache.avaliar(medidores)

        linhas.append({
            "numero_medidores": len(medidores),
            "macro_f1_cv": avaliacao["f1_macro_medio"],
            "desvio_f1_cv": avaliacao["f1_macro_desvio"],
            "medidores": ";".join(medidores),
        })

    df = pd.DataFrame(linhas)

    df = df.drop_duplicates(
        subset="medidores"
    )

    df = df.sort_values(
        ["numero_medidores", "macro_f1_cv"],
        ascending=[True, False],
    )

    caminho = (
        DIRETORIO_RESULTADOS
        / "pareto.csv"
    )

    df.to_csv(
        caminho,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFronteira de Pareto")
    print(df.to_string(index=False))
    print(f"Arquivo salvo: {caminho}\n", flush=True)

    return df


def salvar_grafico_pareto(df):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["numero_medidores"],
        df["macro_f1_cv"],
        s=80,
    )

    plt.plot(
        df["numero_medidores"],
        df["macro_f1_cv"],
    )

    for _, linha in df.iterrows():
        plt.annotate(
            linha["medidores"],
            (
                linha["numero_medidores"],
                linha["macro_f1_cv"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Número de medidores")
    plt.ylabel("Macro-F1")
    plt.title("Fronteira de Pareto")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    caminho = (
        DIRETORIO_RESULTADOS
        / "fronteira_pareto.png"
    )

    plt.savefig(
        caminho,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    print(f"Gráfico salvo: {caminho}", flush=True)


def salvar_todas_avaliacoes(cache):
    linhas = []

    for resultado in (
        cache.avaliador.cache_avaliacoes.values()
    ):
        linhas.append({
            "numero_medidores": resultado["n_medidores"],
            "macro_f1_cv": resultado["f1_macro_medio"],
            "desvio_f1_cv": resultado["f1_macro_desvio"],
            "medidores": ";".join(resultado["medidores"]),
        })

    if linhas:
        df = pd.DataFrame(linhas).sort_values(
            ["numero_medidores", "macro_f1_cv"],
            ascending=[True, False],
        )

        df.to_csv(
            DIRETORIO_RESULTADOS / "todas_avaliacoes.csv",
            index=False,
            encoding="utf-8-sig",
        )


# ============================================================
# Execução do NSGA-II
# ============================================================
def executar_nsga2(cache):
    problema = ProblemaMedidores(cache)

    algoritmo = NSGA2(
        pop_size=TAMANHO_POPULACAO,
        sampling=AmostragemValida(),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(
            prob_var=1.0 / len(problema.indices_disponiveis)
        ),
        repair=RepararSelecao(),
        eliminate_duplicates=True,
    )

    print("\n" + "=" * 70)
    print("NSGA-II")
    print(
        f"População: {TAMANHO_POPULACAO} | "
        f"Gerações: {NUMERO_GERACOES}"
    )
    print(
        f"Representação: {problema.n_var} posições | "
        f"{len(problema.indices_disponiveis)} disponíveis"
    )
    print(
        f"Limite: 1 a {MAXIMO_MEDIDORES} medidores"
    )
    print(
        f"Macro-F1 mínimo: {DESEMPENHO_MINIMO:.2f}"
    )
    print(
        "Crossover: Two Point"
    )
    print(
        "Mutação: Bit Flip | "
        f"p = 1/{len(problema.indices_disponiveis)} "
        f"= {1.0 / len(problema.indices_disponiveis):.4f}"
    )
    print(
        "Objetivos: minimizar N e (1 - Macro-F1 CV)"
    )
    print("=" * 70, flush=True)

    resultado = minimize(
        problema,
        algoritmo,
        termination=("n_gen", NUMERO_GERACOES),
        seed=SEMENTE,
        verbose=True,
    )

    df_pareto = salvar_pareto(
        resultado,
        problema,
    )

    salvar_grafico_pareto(df_pareto)
    salvar_todas_avaliacoes(cache)


# ============================================================
# Execução principal
# ============================================================
if __name__ == "__main__":
    DIRETORIO_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    avaliador = AvaliadorMedidores()
    cache = CacheAvaliacoes(avaliador)

    executar_nsga2(cache)

    print("\nExecução concluída.")
    print(
        f"Novas combinações avaliadas: "
        f"{cache.novas_avaliacoes}"
    )
    print(
        "O conjunto de teste permaneceu reservado."
    )
