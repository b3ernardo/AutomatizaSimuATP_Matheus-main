from pathlib import Path
import re
import numpy as np
import pandas as pd
from f_recortar_janela import recortar_janela
from f_selecionar_sinais_ponto import selecionar_sinais_ponto
from f_preparar_sinais import preparar_sinais
from f_extrair_harmonicas import extrair_harmonicas
from f_calcular_parametros import calcular_parametros
from f_calcular_estatisticas import calcular_estatisticas

# ============================================================
# Configurações
# ============================================================
DIRETORIO_PIPELINE = Path(__file__).resolve().parent
DIRETORIO_PARQUETS = (DIRETORIO_PIPELINE / "Parquets") # pasta com os Parquets
DIRETORIO_DATASETS = (DIRETORIO_PIPELINE / "DatasetAtributos") # pasta onde ficarão os datasets

# Cria o diretório para os datasets caso ainda não exista
DIRETORIO_DATASETS.mkdir(parents=True, exist_ok=True)

# Janelas temporais consideradas (ciclos antes, ciclos depois)
JANELAS = [
    (20, 20), 
    (10, 10), 
    (5, 5),
]

# Configurações da análise harmônica
AMOSTRAS_POR_CICLO = 64
ORDEM_HARMONICA_MAXIMA = 20

# Relação de medidores por região
REGIOES = {
    "R1": [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15],
    "R2": [11, 13, 14, 16, 17, 18, 19, 20, 33, 34],
    "R3": [21, 22, 23, 24, 25, 26, 29, 32],
    "R4": [27, 28, 30, 31],
}

# ============================================================
# Localização dos arquivos Parquet
# ============================================================
arquivos_parquet = sorted(DIRETORIO_PARQUETS.glob("*.parquet"))
total_arquivos = len(arquivos_parquet)

if total_arquivos == 0:
    raise FileNotFoundError(
        f"Nenhum arquivo .parquet encontrado em: "
        f"{DIRETORIO_PARQUETS}"
    )

print(f"Total de Parquets: {total_arquivos}")

# ============================================================
# Processamento
# ============================================================
# Será criado um dataset para cada janela
datasets = {
    janela: []
    for janela in JANELAS
}

# Percorre todos os arquivos .parquet
for indice, arquivo_parquet in enumerate(
    arquivos_parquet,
    start=1,
):
    # Indica o progresso do processamento
    if (indice == 1 or indice % 50 == 0 or indice == total_arquivos):
        print(
            f"Processando {indice}/{total_arquivos}: "
            f"{arquivo_parquet.name}"
        )

    # Busca a fase da falta com base no nome do arquivo
    resultado_fase = re.search(
        r"Fase([ABC])",
        arquivo_parquet.name,
        flags=re.IGNORECASE,
    )

    if resultado_fase is None:
        raise ValueError(
            f"Não foi possível identificar a fase da falta "
            f"em: {arquivo_parquet.name}"
        )

    fase_faltosa = resultado_fase.group(1).upper()

    # Busca a barra da falta com base no nome do arquivo
    resultado_barra = re.search(
        r"barra(\d+)",
        arquivo_parquet.name,
        flags=re.IGNORECASE,
    )

    if resultado_barra is None:
        raise ValueError(
            f"Não foi possível identificar a barra da falta "
            f"em: {arquivo_parquet.name}"
        )

    barra_falta = int(resultado_barra.group(1))

    # Contendo a barra da falta, identifica-se a região da falta
    regiao_falta = None

    for regiao, barras in REGIOES.items():
        if barra_falta in barras:
            regiao_falta = regiao
            break

    if regiao_falta is None:
        raise ValueError(
            f"A barra {barra_falta} não pertence a nenhuma "
            f"região definida. "
            f"Arquivo: {arquivo_parquet.name}"
        )

    # Leitura do .parquet
    df = pd.read_parquet(arquivo_parquet)

    for ciclos_antes, ciclos_depois in JANELAS:
        # Faz o recorte dos dados considerando a janela
        df_janela = recortar_janela(
            df,
            ciclos_antes=ciclos_antes,
            ciclos_depois=ciclos_depois,
        )

        # Nesse primeiro, seleciona apenas os sinais de um ponto específico: SUB
        df_ponto = selecionar_sinais_ponto(
            df_janela,
            "SUB", # deixar dinâmico no futuro
        )

        # Prepara a representação dos sinais enquanto soma das fases ou apenas a fase faltosa
        sinais = preparar_sinais(
            df_ponto,
            fase_faltosa=fase_faltosa,
        )

        # Extração dos atributos da simulação
        atributos_simulacao = {}

        # Primeiro roda com representação = "soma" e depois com representação "fase_faltosa"
        for representacao, grandezas in sinais.items():
            # Extração das harmônicas da tensão
            harmonicas_tensao = extrair_harmonicas(
                grandezas["tensao"],
                amostras_por_ciclo=AMOSTRAS_POR_CICLO,
                ordem_maxima=ORDEM_HARMONICA_MAXIMA,
            )

            # Extração das harmônicas da corrente
            harmonicas_corrente = extrair_harmonicas(
                grandezas["corrente"],
                amostras_por_ciclo=AMOSTRAS_POR_CICLO,
                ordem_maxima=ORDEM_HARMONICA_MAXIMA,
            )

            # Cálculo dos 9 parâmetros: energia, ângulo e módulo (para tensão e corrente),
            # ângulo do fator de potência, impedância, potência ativa
            parametros = calcular_parametros(
                harmonicas_tensao,
                harmonicas_corrente,
            )

            # Cálculo das 6 estatísticas: média, max, desvio padrão, variância, min e amplitude
            atributos = calcular_estatisticas(
                parametros
            )

            # Inclui 'soma' ou 'fase_faltosa' ao nome do atributo
            for nome_atributo, valor in atributos.items():
                nome_completo = (
                    f"{representacao}_"
                    f"{nome_atributo}"
                )

                atributos_simulacao[nome_completo] = valor

        # Cada linha no dataset contém esses metadados
        linha = {
            "arquivo": arquivo_parquet.name,
            "fase_faltosa": fase_faltosa,
            "barra": barra_falta,
            "regiao": regiao_falta,
        }

        # e os 2160 atributos
        linha.update(atributos_simulacao)
        
        datasets[(ciclos_antes, ciclos_depois)].append(linha)

# ============================================================
# Salvamento e validação dos datasets
# ============================================================
print("\n============================================")
print("Salvamento e validação dos datasets")
print("============================================")

for (ciclos_antes, ciclos_depois), linhas in datasets.items():
    # Criação do DataFrame
    df_dataset = pd.DataFrame(linhas)

    # Validação da quantidade de simulações
    if len(df_dataset) != total_arquivos:
        raise ValueError(
            f"Janela [{ciclos_antes}, +{ciclos_depois}]: "
            f"esperadas {total_arquivos} simulações, "
            f"mas foram encontradas {len(df_dataset)}."
        )

    # Verificação de arquivos duplicados
    quantidade_duplicados = (
        df_dataset["arquivo"]
        .duplicated()
        .sum()
    )

    # Seleciona apenas os atributos
    df_atributos = df_dataset.drop(
        columns=[
            "arquivo",
            "fase_faltosa",
            "barra",
            "regiao",
        ]
    )

    # Verificação de NaN
    quantidade_nan = (
        df_atributos
        .isna()
        .sum()
        .sum()
    )

    # Verificação de infinitos
    quantidade_inf = np.isinf(
        df_atributos.to_numpy(
            dtype=float
        )
    ).sum()

    # Se houver problemas, interrompe antes de salvar
    if quantidade_duplicados > 0:
        raise ValueError(
            f"Foram encontrados "
            f"{quantidade_duplicados} arquivos duplicados "
            f"na janela [{ciclos_antes}, +{ciclos_depois}]."
        )

    if quantidade_nan > 0:
        raise ValueError(
            f"Foram encontrados "
            f"{quantidade_nan} valores NaN "
            f"na janela [{ciclos_antes}, +{ciclos_depois}]."
        )

    if quantidade_inf > 0:
        raise ValueError(
            f"Foram encontrados "
            f"{quantidade_inf} valores infinitos "
            f"na janela [{ciclos_antes}, +{ciclos_depois}]."
        )

    # Arquivo de saída
    arquivo_saida = (
        DIRETORIO_DATASETS
        / (
            f"atributos_"
            f"{ciclos_antes}_"
            f"{ciclos_depois}.parquet"
        )
    )

    # Salvamento do parquet com os atributos e por janela
    df_dataset.to_parquet(
        arquivo_saida,
        index=False,
    )

    # Resultados
    print(
        f"\nJanela: "
        f"[-{ciclos_antes}, +{ciclos_depois}]"
    )

    print(
        f"Dataset: {arquivo_saida.name}"
    )

    print(
        f"Simulações: {len(df_dataset)}"
    )

    print(
        f"Atributos: {len(df_atributos.columns)}"
    )

    print(
        f"Colunas totais: {len(df_dataset.columns)}"
    )

    print(
        f"Arquivos duplicados: {quantidade_duplicados}"
    )

    print(
        f"NaN: {quantidade_nan}"
    )

    print(
        f"Inf: {quantidade_inf}"
    )

print("\nProcessamento concluído!")