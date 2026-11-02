import re
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Permite importar as funcoes auxiliares da pasta principal da pipeline
DIRETORIO_PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DIRETORIO_PIPELINE))
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
DIRETORIO_OTIMIZACAO = Path(__file__).resolve().parent
DIRETORIO_PARQUETS = DIRETORIO_PIPELINE / "Parquets16"
DIRETORIO_DATASETS = DIRETORIO_OTIMIZACAO / "DatasetAtributosMedidores16"

# Cria a pasta de saída caso ainda não exista
DIRETORIO_DATASETS.mkdir(
    parents=True,
    exist_ok=True,
)

# Janelas temporais em relação ao TTOQ
JANELAS = [
    # Somente pós-TTOQ (janela de interesse)
    (0, 5),
    # (0, 10),
    # (0, 20),

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
# A lista de pontos será identificada automaticamente a partir dos Parquets.
# Somente pontos com as três tensões e as três correntes serão utilizados.
ATRIBUTOS_DE_INTERESSE = [
    "soma_modulo_corrente_H1_maximo",
    "soma_modulo_corrente_H1_media",
    "soma_modulo_corrente_H1_minimo",
    "soma_modulo_tensao_H1_maximo",
    "soma_modulo_tensao_H1_media",
    "soma_modulo_tensao_H1_minimo",
]

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
# Identificação dos pontos de medição disponíveis
# ============================================================
# Lê o primeiro Parquet para identificar os pontos existentes.
df_referencia = pd.read_parquet(arquivos_parquet[0])

pontos_encontrados = sorted({
    resultado.group(1)
    for coluna in df_referencia.columns
    if "(V-node)" in coluna
    for resultado in [re.match(r"^(M\d{2})", coluna)]
    if resultado is not None
})

pontos_encontrados = ["SUB"] + pontos_encontrados

# Confirma a presença das fases A, B e C em tensão e corrente.
def validar_sinais_ponto(df, ponto):
    df_ponto = selecionar_sinais_ponto(df, ponto)

    colunas_tensao = [
        coluna for coluna in df_ponto.columns
        if "(V-node)" in coluna
    ]
    colunas_corrente = [
        coluna for coluna in df_ponto.columns
        if "(I-bran)" in coluna
    ]

    fases_tensao = {
        coluna.split(" (")[0].rstrip("-")[-1]
        for coluna in colunas_tensao
    }
    fases_corrente = {
        coluna.split(" (")[0].split("-")[-1][-1]
        for coluna in colunas_corrente
    }

    if (
        len(colunas_tensao) != 3
        or len(colunas_corrente) != 3
        or fases_tensao != {"A", "B", "C"}
        or fases_corrente != {"A", "B", "C"}
    ):
        raise ValueError(
            f"O ponto {ponto} não possui as três tensões "
            "e as três correntes necessárias."
        )

    return df_ponto

# Não inclui na otimização pontos com medições incompletas.
pontos_medicao = []
pontos_excluidos = []

for ponto in pontos_encontrados:
    try:
        validar_sinais_ponto(df_referencia, ponto)
        pontos_medicao.append(ponto)
    except ValueError as erro:
        pontos_excluidos.append((ponto, str(erro)))

if not pontos_medicao:
    raise ValueError("Nenhum ponto de medição completo foi encontrado.")

print(f"Pontos de medição válidos ({len(pontos_medicao)}): {pontos_medicao}")

if pontos_excluidos:
    print(f"Pontos excluídos ({len(pontos_excluidos)}): {pontos_excluidos}")

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

        # Extrai os seis atributos H1 de cada ponto de medição válido.
        atributos_simulacao = {}

        for ponto in pontos_medicao:
            # Garante que as seis medições também existam neste Parquet.
            df_ponto = validar_sinais_ponto(df_janela, ponto)

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
            atributos_ponto = {}

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

                for nome_atributo, valor in atributos.items():
                    nome_completo = (
                        f"{representacao}_"
                        f"{nome_atributo}"
                    )

                    # Mantém somente os seis atributos usados no classificador.
                    if nome_completo in ATRIBUTOS_DE_INTERESSE:
                        atributos_ponto[nome_completo] = valor

            # Evita salvar um ponto com atributos ausentes.
            atributos_ausentes = (
                set(ATRIBUTOS_DE_INTERESSE) - set(atributos_ponto)
            )

            if atributos_ausentes:
                raise ValueError(
                    f"Atributos ausentes para {ponto} "
                    f"em {arquivo_parquet.name}: "
                    f"{sorted(atributos_ausentes)}"
                )

            # Prefixa os atributos para identificar seu medidor.
            for nome_atributo in ATRIBUTOS_DE_INTERESSE:
                atributos_simulacao[
                    f"{ponto}_{nome_atributo}"
                ] = atributos_ponto[nome_atributo]

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

    # Verifica se existem exatamente seis atributos por ponto.
    quantidade_atributos_esperada = 6 * len(pontos_medicao)
    quantidade_atributos_obtida = len(df_dataset.columns) - 3

    if quantidade_atributos_obtida != quantidade_atributos_esperada:
        raise ValueError(
            f"Esperados {quantidade_atributos_esperada} atributos, "
            f"mas foram encontrados {quantidade_atributos_obtida}."
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