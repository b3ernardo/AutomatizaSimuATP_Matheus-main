from pathlib import Path
import re
import pandas as pd

# ============================================================
# Diretórios
# ============================================================
DIRETORIO_PIPELINE = Path(__file__).resolve().parent

DIRETORIO_CARTOES = (
    DIRETORIO_PIPELINE
    / "TodosCartoesRodadosSemRetirarCargas"
    / "CartoesATP"
)

# ============================================================
# Localização dos cartões ATP
# ============================================================
arquivos_atp = sorted(DIRETORIO_CARTOES.glob("*.atp"))

print(f"Diretório dos cartões: {DIRETORIO_CARTOES}")
print(f"Total de cartões ATP encontrados: {len(arquivos_atp)}")

# ============================================================
# Seleção dos casos P1/C1
# ============================================================
arquivos_p1_c1 = [
    arquivo
    for arquivo in arquivos_atp
    if "_p1_carregamento1_" in arquivo.stem.lower()
]

print(f"Total de cartões P1/C1 encontrados: {len(arquivos_p1_c1)}")

# ============================================================
# Extração dos metadados a partir do nome do arquivo
# ============================================================
padrao_nome = re.compile(
    r"^Fase(?P<fase>[ABC])_"
    r"(?P<gd>comGD|semGD)_"
    r"p(?P<p>\d+)_"
    r"carregamento(?P<carregamento>\d+)_"
    r"barra(?P<barra>\d+)"
    r"solo(?P<solo>\d+)"
    r"retirado(?P<retirada>.+?)"
    r"fai(?P<fai>\d+)$",
    re.IGNORECASE
)

registros = []
arquivos_invalidos = []

for arquivo in arquivos_p1_c1:
    correspondencia = padrao_nome.match(arquivo.stem)

    if correspondencia is None:
        arquivos_invalidos.append(arquivo.name)
        continue

    dados = correspondencia.groupdict()

    registros.append(
        {
            "arquivo": arquivo.name,
            "fase": dados["fase"].upper(),
            "gd": dados["gd"],
            "p": int(dados["p"]),
            "carregamento": int(dados["carregamento"]),
            "barra": int(dados["barra"]),
            "solo": int(dados["solo"]),
            # "retirada": dados["retirada"],
            "fai": int(dados["fai"]),
        }
    )

if arquivos_invalidos:
    print("\nATENÇÃO: arquivos que não seguiram o padrão esperado:")
    for arquivo in arquivos_invalidos:
        print(f"  - {arquivo}")
else:
    print("\nTodos os cartões P1/C1 seguiram o padrão esperado.")

# ============================================================
# Criação da tabela
# ============================================================
df_casos = pd.DataFrame(registros)

print("\nMetadados extraídos com sucesso.")
print(f"Total de registros: {len(df_casos)}")

print("\nPrimeiros registros:")
print(df_casos.head())

# ============================================================
# Análise dos cenários
# ============================================================
print("\nValores encontrados por variável:")

for coluna in ["fase", "gd", "barra", "solo", "fai"]:
    valores = sorted(df_casos[coluna].unique().tolist())
    print(f"{coluna}: {valores}")

# ============================================================
# Distribuição dos casos
# ============================================================
print("\nQuantidade de casos por condição de GD:")
print(df_casos.groupby("gd").size())

print("\nQuantidade de casos por fase:")
print(df_casos.groupby("fase").size())

print("\nQuantidade de casos por barra:")
print(df_casos.groupby("barra").size().sort_index())