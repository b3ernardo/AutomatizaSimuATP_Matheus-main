import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# ============================================================
# CONFIGURAÇÕES
# ============================================================
DIRETORIO_OTIMIZACAO = Path(__file__).resolve().parent
# Diretório do pipeline original
DIRETORIO_PIPELINE = DIRETORIO_OTIMIZACAO.parent
# Diretório para salvar os resultados das execuções
DIRETORIO_RESULTADOS = DIRETORIO_OTIMIZACAO / "Resultados"

sys.path.insert(0, str(DIRETORIO_PIPELINE))

from Regionalizacoes import REGIONALIZACOES

NUMERO_REGIOES = 8 # número de regiões
NOME_JANELA = "0/5" # janela
PROPORCAO_TESTE = 0.20 # proporção destinada ao conjunto de teste
SEMENTE = 42 # semente para reprodutibilidade
NUMERO_FOLDS = 5 # número de folds da validação cruzada

TAMANHO_POPULACAO = 100 # tamanho da população do NSGA-II
NUMERO_GERACOES = 100 # número de gerações do NSGA-II

# Caminho do parquet com os dados dos atributos para cada medidor
CAMINHO_DATASET = DIRETORIO_OTIMIZACAO / "DatasetAtributosMedidores16" / "atributos_0_5.parquet"
# Caminho do json com os hiperparâmetros otimizados para a RF
CAMINHO_HIPERPARAMETROS = DIRETORIO_PIPELINE / "Hiperparametros" / "hiperparametros_rf_16.json"
# Caminho do json do cache de avaliações do RF
CAMINHO_CACHE = DIRETORIO_OTIMIZACAO / "CacheAvaliacoes" / "cache_avaliacoes.json"

# Medidores que podem ser utilizados nas configurações
MEDIDORES_DISPONIVEIS = [
    "SUB", "M02", "M03", "M04", "M05", "M06", "M07", "M08",
    "M11", "M14", "M16", "M17", "M19", "M20", "M21", "M22",
    "M24", "M25", "M26", "M27", "M28", "M29", "M32", "M33", 
    "M34"
]

# Atributos definidos na etapa de seleção
ATRIBUTOS_DE_INTERESSE = [
    "soma_modulo_corrente_H1_maximo",
    "soma_modulo_corrente_H1_media",
    "soma_modulo_corrente_H1_minimo",
    "soma_modulo_tensao_H1_maximo",
    "soma_modulo_tensao_H1_media",
    "soma_modulo_tensao_H1_minimo",
]

# ============================================================
# CACHE DAS AVALIAÇÕES
# ============================================================
def carregar_cache():
    # Se o arquivo ainda não existe, inicia com um cache vazio
    if not CAMINHO_CACHE.is_file():
        return {}

    # Carrega as avaliações realizadas em execuções anteriores
    with CAMINHO_CACHE.open("r", encoding="utf-8") as arquivo:
        return json.load(arquivo)

def salvar_cache(cache):
    # Cria a pasta do cache caso ela ainda não exista
    CAMINHO_CACHE.parent.mkdir(parents=True, exist_ok=True)

    # Salva o cache atualizado
    with CAMINHO_CACHE.open("w", encoding="utf-8") as arquivo:
        json.dump(cache, arquivo, indent=4, ensure_ascii=False)

# ============================================================
# CARREGAMENTO DOS HIPERPARÂMETROS
# ============================================================
def carregar_hiperparametros():
    if not CAMINHO_HIPERPARAMETROS.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {CAMINHO_HIPERPARAMETROS}")

    # Abre o arquivo JSON e transforma seu conteúdo em um objeto Python
    with CAMINHO_HIPERPARAMETROS.open("r", encoding="utf-8") as arquivo:
        cache = json.load(arquivo)

    chave = f"{NOME_JANELA}|{NUMERO_REGIOES}" # "0/5|8"

    if chave not in cache:
        raise ValueError(f"Hiperparâmetros não encontrados para '{chave}'.")

    # Pega os hiperparâmetros para a respectiva janela e quantidade de regiões
    return cache[chave]["parametros"]

# ============================================================
# DECODIFICAÇÃO DA CONFIGURAÇÃO DE MEDIDORES
# ============================================================
# Decodifica o vetor binário x para obter os medidores selecionados
def decodificar_medidores(x):
    medidores = []

    # Supondo MEDIDORES_DISPONIVEIS = ["SUB", "M02", "M03", "M04", "M05"] e x = [0, 1, 0, 1, 0], o zip faz:
    # ("SUB", 0)
    # ("M02", 1)
    # ("M03", 0)
    # ("M04", 1)
    # ("M05", 0)
    for medidor, selecionado in zip(MEDIDORES_DISPONIVEIS, x):
        if selecionado:
            medidores.append(medidor)

    # x = [0, 1, 0, 1, 0] -> medidores = ["M02", "M04"]
    return medidores

# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================
def preparar_dados():
    if not CAMINHO_DATASET.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {CAMINHO_DATASET}")

    # Lê o parquet com os atributos de cada medidor
    df = pd.read_parquet(CAMINHO_DATASET)

    # --------------------------------------------------------
    # Split treino/teste
    # --------------------------------------------------------
    # Para cada barra, 80% das simulações (arquivos) vão para treinamento e 20% para teste
    arquivos_treino, arquivos_teste = train_test_split(
        df["arquivo"],
        test_size=PROPORCAO_TESTE, # define 80/20
        random_state=SEMENTE, # fixa a aleatoriedade, garantindo que a divisão seja sempre a mesma
        stratify=df["barra"],
    )

    # Pega só a parte de treinamento
    df_treino = df.set_index("arquivo").loc[list(arquivos_treino)]

    # --------------------------------------------------------
    # Regionalização
    # --------------------------------------------------------
    regioes = REGIONALIZACOES[NUMERO_REGIOES] # busca a regionalização
    mapa_barra_regiao = {}

    # Mapeia qual é a região de cada barra: mapa_barra_regiao = {2: "R1", 3: "R1", ..., 31: "R8"}
    for nome_regiao, barras in regioes.items():
        for barra in barras:
            if barra in mapa_barra_regiao:
                raise ValueError(f"A barra {barra} pertence a mais de uma região.")

            mapa_barra_regiao[barra] = nome_regiao

    # --------------------------------------------------------
    # Variável resposta: região de ocorrência da falta
    # --------------------------------------------------------
    # Pega, para cada simulação de treinamento, a barra onde a falta ocorreu
    barras_treino = df_treino["barra"].astype(int)

    # Substitui cada número de barra pela região correspondente
    y_treino = barras_treino.map(mapa_barra_regiao)

    if y_treino.isna().any():
        raise ValueError("Existem barras sem região definida.")

    if y_treino.nunique() != NUMERO_REGIOES:
        raise ValueError("Nem todas as regiões estão presentes no treinamento.")

    # --------------------------------------------------------
    # 5-fold Cross-Validation
    # --------------------------------------------------------
    # Divide as simulações de treinamento em k partes
    validacao = StratifiedKFold(
        n_splits=NUMERO_FOLDS,
        shuffle=True,
        random_state=SEMENTE,
    )

    # Gera os índices de treinamento e validação de cada partição
    # Em cada partição, 4/5 fica para treinamento e 1/5 para validação
    # O StratifiedKFold ajuda a manter a proporção de barras em cada partição
    particoes = list(validacao.split(df_treino, barras_treino))

    # --------------------------------------------------------
    # Hiperparâmetros da Random Forest
    # --------------------------------------------------------
    hiperparametros = carregar_hiperparametros()

    # --------------------------------------------------------
    # Informações
    # --------------------------------------------------------
    print("=" * 70)
    print("BASE DE AVALIAÇÃO")
    print("=" * 70)
    print(f"Simulações totais: {len(df)}")
    print(f"Treinamento: {len(arquivos_treino)}")
    print(f"Teste reservado: {len(arquivos_teste)}")
    print(f"Regiões: {NUMERO_REGIOES}")
    print(f"Validação cruzada: {NUMERO_FOLDS} folds")
    print(f"Hiperparâmetros: {NOME_JANELA}|{NUMERO_REGIOES}\n")

    return df_treino, y_treino, particoes, hiperparametros, sorted(regioes)

# ============================================================
# AVALIAÇÃO DE UMA CONFIGURAÇÃO DE MEDIDORES
# ============================================================
def avaliar_configuracao(medidores, df_treino, y_treino, particoes, hiperparametros, regioes):
    if len(medidores) == 0:
        raise ValueError("É necessário selecionar pelo menos um medidor.")

    if len(medidores) != len(set(medidores)):
        raise ValueError("Há medidores repetidos.")

    # --------------------------------------------------------
    # Seleção dos atributos
    # --------------------------------------------------------
    # Busca os seis atributos referentes a cada medidor selecionado
    colunas = [
        f"{medidor}_{atributo}"
        for medidor in medidores
        for atributo in ATRIBUTOS_DE_INTERESSE
    ]

    colunas_ausentes = set(colunas) - set(df_treino.columns)

    if colunas_ausentes:
        raise ValueError("Atributos não encontrados:\n" + "\n".join(sorted(colunas_ausentes)))

    # Define as variáveis de entrada conforme os atributos dos medidores selecionados
    X = df_treino[colunas].to_numpy(dtype=float)

    # Conjunto de variável resposta
    y = y_treino.to_numpy()

    if not np.isfinite(X).all():
        raise ValueError("Existem valores NaN ou infinitos nos atributos.")

    # --------------------------------------------------------
    # Avaliação nas cinco partições
    # --------------------------------------------------------
    resultados_particoes = []

    # Para cada partição
    for indices_treino, indices_validacao in particoes:
        modelo = RandomForestClassifier(random_state=SEMENTE, n_jobs=-1)

        # Define os hiperparâmetros da RF
        modelo.set_params(**hiperparametros)

        # Treina o modelo da RF com os dados de treino da partição atual
        modelo.fit(X[indices_treino], y[indices_treino])

        # Realiza a predição com os dados de validação da partição atual
        y_predito = modelo.predict(X[indices_validacao])

        # Calcula o Macro-F1 da partição atual
        # Macro-F1 = (F1_R1 + F1_R2 + ... + F1_R8) / 8
        macro_f1 = f1_score(
            y[indices_validacao],
            y_predito,
            average="macro",
            labels=regioes,
            zero_division=0,
        )

        resultados_particoes.append(float(macro_f1))

    # --------------------------------------------------------
    # Resultado
    # --------------------------------------------------------
    return {
        "medidores": medidores,
        "numero_medidores": len(medidores),
        "numero_atributos": len(colunas),
        "macro_f1_medio": float(np.mean(resultados_particoes)),
        "macro_f1_desvio": float(np.std(resultados_particoes)),
        "macro_f1_particoes": resultados_particoes,
    }
    
# ============================================================
# SALVAMENTO DOS RESULTADOS
# ============================================================
def salvar_resultados(resultado, problema):
    DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)

    solucoes_finais = []

    # Percorre as soluções da fronteira final
    for x, f in zip(resultado.X, resultado.F):
        medidores = decodificar_medidores(x)

        solucoes_finais.append({
            "medidores": medidores,
            "numero_medidores": len(medidores),
            "macro_f1": float(1 - f[1]),
            "f1": int(f[0]),
            "f2": float(f[1]),
        })

    dados = {
        "semente": SEMENTE,
        "tamanho_populacao": TAMANHO_POPULACAO,
        "numero_geracoes": NUMERO_GERACOES,
        "numero_regioes": NUMERO_REGIOES,
        "janela": NOME_JANELA,
        "numero_folds": NUMERO_FOLDS,
        "avaliacoes_rf": problema.numero_avaliacoes_rf,
        "avaliacoes_cache": problema.numero_avaliacoes_cache,
        "solucoes_finais": solucoes_finais,
    }

    caminho = DIRETORIO_RESULTADOS / f"execucao_seed_{SEMENTE}.json"

    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, indent=4, ensure_ascii=False)

# ============================================================
# DEFINIÇÃO DO PROBLEMA MULTIOBJETIVO
# ============================================================
# Cada posição do vetor binário representa um medidor de MEDIDORES_DISPONIVEIS
class ProblemaMedidores(ElementwiseProblem):
    def __init__(self, df_treino, y_treino, particoes, hiperparametros, regioes):
        super().__init__(
            n_var=len(MEDIDORES_DISPONIVEIS), # número de variáveis de decisão do problema
            n_obj=2, # quantas funções objetivo o problema possui
            n_ieq_constr=1, # número de restrições de desigualdade
            xl=0, # limite inferior das variáveis
            xu=1, # limite superior das variáveis
            vtype=bool, # tipo das variáveis de decisão (booleanas/binárias)
        )

        self.df_treino = df_treino
        self.y_treino = y_treino
        self.particoes = particoes
        self.hiperparametros = hiperparametros
        self.regioes = regioes
        self.cache = carregar_cache()
        self.numero_avaliacoes_rf = 0
        self.numero_avaliacoes_cache = 0

    def _evaluate(self, x, out, *args, **kwargs):
        # Converte o vetor binário nos medidores selecionados
        # Considerando ["SUB", "M02", "M03", "M04", "M05"]: [0, 1, 0, 1, 0] -> ["M02", "M04"]
        medidores = decodificar_medidores(x) # nesse problema, x tem sempre tamanho 25

        # Restrição de desigualdade: pelo menos um medidor deve ser selecionado
        # Se len(medidores) = 2, temos: 1 - 2 = -1 <= 0 (válido)
        # Se len(medidores) = 0, temos: 1 - 0 = 1 > 0 (inválido)
        out["G"] = [1 - len(medidores)] # armazena os valores das restrições

        # Se nenhum medidor foi selecionado, não executa a Random Forest
        if len(medidores) == 0:
            out["F"] = [0, 1]
            return

        # Usa o vetor binário como chave do cache: [0, 1, 0, 1, 0] -> "01010"
        chave = "".join("1" if valor else "0" for valor in x)

        # Se a configuração já foi avaliada, reutiliza o resultado
        if chave in self.cache:
            resultado = self.cache[chave]
            self.numero_avaliacoes_cache += 1
        else:
            # Avalia a configuração utilizando a RF e 5-fold Cross-Validation
            resultado = avaliar_configuracao(
                medidores, # configuração, por ex.: ["M02", "M04"]
                self.df_treino,
                self.y_treino,
                self.particoes,
                self.hiperparametros,
                self.regioes,
            )

            self.numero_avaliacoes_rf += 1
            
            # Salva o resultado no cache
            self.cache[chave] = resultado
            salvar_cache(self.cache)

        # Objetivo 1: minimizar o número de medidores
        f1 = len(medidores)

        # Objetivo 2: minimizar o erro de classificação (ou maximizar o Macro-F1)
        f2 = 1 - resultado["macro_f1_medio"]

        out["F"] = [f1, f2] # armazena os valores de f1 e f2
   
# ============================================================
# EXECUÇÃO DO NSGA-II
# ============================================================
if __name__ == "__main__":
    # Retorna a base de treinamento, variável resposta, partições, hiperparâmetros usados e regionalização
    df_treino, y_treino, particoes, hiperparametros, regioes = preparar_dados()

    # Inicializa o problema
    problema = ProblemaMedidores(df_treino, y_treino, particoes, hiperparametros, regioes)

    # --------------------------------------------------------
    # Configuração do NSGA-II
    # --------------------------------------------------------
    # BinaryRandomSampling: gera a população inicial binária aleatoriamente
    # Considerando ["SUB", "M02", "M03", "M04"]:
    # ex.: [0, 1, 0, 1] -> ["M02", "M04"]

    # UniformCrossover: combina dois pais posição a posição, sendo que cada posição
    # possui a mesma probabilidade de participar da troca
    # ex.: [1, 1, 0, 0] + [0, 0, 1, 1] -> [1, 0, 1, 0]

    # BitflipMutation: pode inverter bits do indivíduo, adicionando ou removendo medidores
    # ex.: [1, 0, 1, 0] -> [1, 0, 0, 0]
    algoritmo = NSGA2(
        pop_size=TAMANHO_POPULACAO,
        sampling=BinaryRandomSampling(),
        crossover=UniformCrossover(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True, # elimina indivíduos com vetores binários idênticos
    )

    # --------------------------------------------------------
    # Execução da otimização
    # --------------------------------------------------------
    resultado = minimize(
        problema, # problema multiobjetivo que será otimizado
        algoritmo, # NSGA-II
        ("n_gen", NUMERO_GERACOES), # critério de parada: encerra após atingir o número definido de gerações
        seed=SEMENTE, # fixa a semente aleatória para permitir a reprodução da execução
        verbose=True, # exibe no terminal as informações de progresso da otimização
    )
        
    print(f"\nAvaliações realizadas pela RF: {problema.numero_avaliacoes_rf}")
    print(f"Resultados recuperados do cache: {problema.numero_avaliacoes_cache}")
    
    salvar_resultados(resultado, problema)

    Scatter().add(resultado.F).show()