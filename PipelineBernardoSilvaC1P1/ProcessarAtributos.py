import re
import numpy as np
import pandas as pd
from pathlib import Path
from f_recortar_janela import recortar_janela
from f_selecionar_sinais_ponto import selecionar_sinais_ponto
from f_preparar_sinais import preparar_sinais
from f_extrair_harmonicas import extrair_harmonicas
from f_calcular_parametros import calcular_parametros
from f_calcular_estatisticas import calcular_estatisticas

# ============================================================
# Configurações
# ============================================================
# Diretórios 
DIRETORIO_PIPELINE = Path(__file__).resolve().parent
DIRETORIO_PARQUETS = DIRETORIO_PIPELINE / "Parquets16"
DIRETORIO_DATASETS = DIRETORIO_PIPELINE / "DatasetAtributos16"

# Cria a pasta de saída caso ainda não exista
DIRETORIO_DATASETS.mkdir(
    parents=True,
    exist_ok=True,
)

# Janelas temporais em relação ao TTOQ
JANELAS = [
    # Somente pós-TTOQ (janela de interesse)
    (0, 5),
    (0, 10),
    (0, 20),

    # Pré e pós-TTOQ
    # (5, 5),
    # (10, 10),
    # (20, 20),

    # Somente pré-TTOQ
    # (5, 0),
    # (10, 0),
    # (20, 0),
]

# Parâmetros utilizados pelas funções auxiliares de extração harmônica
AMOSTRAS_POR_CICLO = 16
ORDEM_HARMONICA_MAXIMA = 1
PONTO_MEDICAO = "SUB"

# ============================================================
# Localização dos arquivos Parquet
# ============================================================
# Lista os arquivos Parquet presentes na pasta /Parquets
arquivos_parquet = sorted(DIRETORIO_PARQUETS.glob("*.parquet"))
total_arquivos = len(arquivos_parquet)

if total_arquivos == 0:
    raise FileNotFoundError(
        "Nenhum arquivo .parquet encontrado em: "
        f"{DIRETORIO_PARQUETS}"
    )

print(f"Total de Parquets: {total_arquivos}")

# ============================================================
# Estrutura dos datasets
# ============================================================
# Dicionário para armazenar os atributos extraídos de cada janela temporal
datasets = {
    janela: []
    for janela in JANELAS
}

# ============================================================
# Processamento
# ============================================================
# Cada arquivo corresponde a uma simulação e seus sinais são lidos uma vez
for indice, arquivo_parquet in enumerate(
    arquivos_parquet,
    start=1,
):
    # Print sobre o status do processamento
    if (
        indice == 1
        or indice % 50 == 0
        or indice == total_arquivos
    ):
        print(
            f"Processando {indice}/{total_arquivos}: "
            f"{arquivo_parquet.name}"
        )

    # --------------------------------------------------------
    # Identificação e extração da fase faltosa
    # --------------------------------------------------------
    resultado_fase = re.search(
        r"Fase([ABC])",
        arquivo_parquet.name,
        flags=re.IGNORECASE,
    )

    if resultado_fase is None:
        raise ValueError(
            "Não foi possível identificar a fase da falta em: "
            f"{arquivo_parquet.name}"
        )

    fase_faltosa = (
        resultado_fase
        .group(1)
        .upper()
    )

    # --------------------------------------------------------
    # Identificação da barra da falta
    # --------------------------------------------------------
    # A barra é o rótulo utilizado posteriormente na classificação
    resultado_barra = re.search(
        r"barra(\d+)",
        arquivo_parquet.name,
        flags=re.IGNORECASE,
    )

    if resultado_barra is None:
        raise ValueError(
            "Não foi possível identificar a barra da falta em: "
            f"{arquivo_parquet.name}"
        )

    barra_falta = int(resultado_barra.group(1))

    # --------------------------------------------------------
    # Leitura dos sinais no Parquet
    # --------------------------------------------------------
    df = pd.read_parquet(arquivo_parquet)

    # --------------------------------------------------------
    # Processamento de cada janela
    # --------------------------------------------------------
    for ciclos_antes, ciclos_depois in JANELAS:
        # Recorte temporal em relação ao TTOQ
        df_janela = recortar_janela(
            df,
            ciclos_antes=ciclos_antes,
            ciclos_depois=ciclos_depois,
        )

        # Seleciona os sinais do ponto de medição utilizado
        df_ponto = selecionar_sinais_ponto(
            df_janela,
            PONTO_MEDICAO, # SUB
        )

        # ----------------------------------------------------
        # Preparação das representações dos sinais
        # ----------------------------------------------------
        sinais = preparar_sinais(
            df_ponto,
            fase_faltosa=fase_faltosa,
        )

        # ----------------------------------------------------
        # Extração dos atributos a partir dos sinais de V e I
        # ----------------------------------------------------
        atributos_simulacao = {}

        for representacao, grandezas in sinais.items():
            # Extrai componentes harmônicas dos sinais de tensão na janela
            harmonicas_tensao = extrair_harmonicas(
                grandezas["tensao"],
                amostras_por_ciclo=AMOSTRAS_POR_CICLO,
                ordem_maxima=ORDEM_HARMONICA_MAXIMA,
            )

            # Repete a extração para os sinais de corrente
            harmonicas_corrente = extrair_harmonicas(
                grandezas["corrente"],
                amostras_por_ciclo=AMOSTRAS_POR_CICLO,
                ordem_maxima=ORDEM_HARMONICA_MAXIMA,
            )

            # Combina as componentes harmônicas em parâmetros elétricos
            parametros = calcular_parametros(
                harmonicas_tensao,
                harmonicas_corrente,
            )

            # Resume os parâmetros por estatísticas, como máximo, média e mínimo
            atributos = calcular_estatisticas(parametros)

            # Prefixa cada coluna com sua representação para evitar ambiguidades
            for nome_atributo, valor in atributos.items():
                nome_completo = (
                    f"{representacao}_"
                    f"{nome_atributo}"
                )

                atributos_simulacao[nome_completo] = valor

        # ----------------------------------------------------
        # Montagem da linha do dataset
        # ----------------------------------------------------
        linha = {
            "arquivo": arquivo_parquet.name,
            "fase_faltosa": fase_faltosa,
            "barra": barra_falta,
        }

        linha.update(atributos_simulacao)

        datasets[(ciclos_antes, ciclos_depois)].append(linha)

# ============================================================
# Salvamento e validação dos datasets
# ============================================================
for (
    ciclos_antes,
    ciclos_depois,
), linhas in datasets.items():
    # Consolida as simulações processadas para a janela corrente
    df_dataset = pd.DataFrame(linhas)

    # --------------------------------------------------------
    # Validação da quantidade de simulações
    # --------------------------------------------------------
    if len(df_dataset) != total_arquivos:
        raise ValueError(
            f"Janela [-{ciclos_antes}, +{ciclos_depois}]: "
            f"esperadas {total_arquivos} simulações, "
            f"mas foram encontradas {len(df_dataset)}."
        )

    # --------------------------------------------------------
    # Verificação de duplicatas
    # --------------------------------------------------------
    # Impede que um mesmo nome de arquivo apareça duas vezes no dataset
    quantidade_duplicados = (
        df_dataset["arquivo"]
        .duplicated()
        .sum()
    )

    # --------------------------------------------------------
    # Seleção exclusiva das colunas de atributos
    # --------------------------------------------------------
    # As verificações numéricas não incluem as colunas de metadados
    df_atributos = df_dataset.drop(
        columns=[
            "arquivo",
            "fase_faltosa",
            "barra",
        ]
    )

    # --------------------------------------------------------
    # Validação dos atributos
    # --------------------------------------------------------
    quantidade_nan = (
        df_atributos
        .isna()
        .sum()
        .sum()
    )

    quantidade_inf = np.isinf(
        df_atributos.to_numpy(
            dtype=float
        )
    ).sum()

    if quantidade_duplicados > 0:
        raise ValueError(
            f"Foram encontrados {quantidade_duplicados} "
            "arquivos duplicados na janela "
            f"[-{ciclos_antes}, +{ciclos_depois}]."
        )

    if quantidade_nan > 0:
        raise ValueError(
            f"Foram encontrados {quantidade_nan} "
            "valores NaN na janela "
            f"[-{ciclos_antes}, +{ciclos_depois}]."
        )

    if quantidade_inf > 0:
        raise ValueError(
            f"Foram encontrados {quantidade_inf} "
            "valores infinitos na janela "
            f"[-{ciclos_antes}, +{ciclos_depois}]."
        )

    # --------------------------------------------------------
    # Salvamento
    # --------------------------------------------------------
    arquivo_saida = (
        DIRETORIO_DATASETS
        / (
            f"atributos_"
            f"{ciclos_antes}_"
            f"{ciclos_depois}.parquet"
        )
    )

    # Preserva os metadados e todas as colunas calculadas no Parquet final
    df_dataset.to_parquet(
        arquivo_saida,
        index=False,
    )

    # --------------------------------------------------------
    # Resumo
    # --------------------------------------------------------
    print(f"\nJanela: [-{ciclos_antes}, +{ciclos_depois}]")
    print(f"Dataset: {arquivo_saida.name}")
    print(f"Simulações: {len(df_dataset)}")
    print(f"Atributos: {len(df_atributos.columns)}")
    print(f"Colunas totais: {len(df_dataset.columns)}")
    print(f"Arquivos duplicados: {quantidade_duplicados}")
    print(f"NaN: {quantidade_nan}")
    print(f"Inf: {quantidade_inf}")

print("\nProcessamento concluído!")