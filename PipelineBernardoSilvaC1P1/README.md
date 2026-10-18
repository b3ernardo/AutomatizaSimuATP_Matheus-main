# Pipeline de processamento e classificação de faltas de alta impedância

Este guia explica como preparar o ambiente no **Windows / Visual Studio Code**, acessar a pasta `PipelineBernardoSilvaC1P1` e executar o script **`ProcessarAtributos.py`**. Também apresenta o papel dos principais scripts e módulos auxiliares.

> **Execução mais rápida:** se a pasta `Parquets` já contém os sinais processados e o ambiente Python está configurado, abra o terminal na pasta `PipelineBernardoSilvaC1P1` e execute `python .\ProcessarAtributos.py`. **Não é necessário executar o ATP novamente** para extrair os atributos de Parquets já existentes.

## 1. Pré-requisitos

- **Windows** e **Visual Studio Code** (com a extensão **Python**, da Microsoft).
- **Python 3** instalado e disponível no terminal. Recomenda-se utilizar uma versão compatível com as bibliotecas do projeto e com `f_readpl4.py`; a versão exata ainda precisa ser confirmada no ambiente original.
- A pasta completa do repositório, incluindo `PipelineBernardoSilvaC1P1`, os arquivos `f_*.py` e os módulos `f_readpl4.py` / `f_processar_pl4.py` na pasta superior.
- Para **executar novas simulações ATP** com `GerarParquet.py`: o executável `runATP.exe` deve estar na pasta superior à `PipelineBernardoSilvaC1P1`, conforme o caminho definido no código; também é necessário que o ATP esteja corretamente instalado/configurado.
- Para **apenas executar `ProcessarAtributos.py`**: os arquivos `.parquet` de entrada devem estar disponíveis em `PipelineBernardoSilvaC1P1/Parquets/`. O ATP e o `runATP.exe` não são necessários nessa etapa.

## 2. Estrutura de pastas

Estrutura simplificada, conforme o projeto apresentado:

```text
AutomatizaSimuATP_Matheus-main/
├── runATP.exe                          # Necessário somente para gerar novos Parquets via ATP
├── f_readpl4.py                        # Leitura dos arquivos PL4 (verificar localização)
├── f_processar_pl4.py                  # Processamento/reamostragem dos sinais (verificar localização)
└── PipelineBernardoSilvaC1P1/
    ├── GerarParquet.py
    ├── ProcessarAtributos.py
    ├── SelecaoAtributos.py
    ├── Classificadores.py
    ├── Regionalizacoes.py
    ├── f_recortar_janela.py
    ├── f_selecionar_sinais_ponto.py
    ├── f_selecionar_sinais_medidores.py
    ├── f_preparar_sinais.py
    ├── f_extrair_harmonicas.py
    ├── f_calcular_parametros.py
    ├── f_calcular_estatisticas.py
    ├── Parquets/                      # Entrada do ProcessarAtributos.py
    ├── DatasetAtributos/              # Saída do ProcessarAtributos.py
    ├── ParquetsDiagnostico/           # Dados de diagnóstico, se utilizados
    ├── temp_atp/                      # Arquivos temporários das simulações
    └── TodosCartoesRodadosSemRetirarCargas/
        └── CartoesATP/                 # Cartões .atp usados pelo GerarParquet.py
```

**Atenção:** a imagem mostra a pasta do projeto, mas não confirma a localização real de `runATP.exe`, `f_readpl4.py` e `f_processar_pl4.py`. Os caminhos acima seguem o que o `GerarParquet.py` fornecido espera; confira antes de executar novas simulações.

## 3. Abrir o projeto e configurar o terminal

1. Abra o **VS Code**.
2. Use **Arquivo > Abrir Pasta** e selecione `PipelineBernardoSilvaC1P1` (ou abra o repositório e navegue até essa subpasta).
3. Abra **Terminal > Novo Terminal**.
4. Confirme que o terminal está na pasta `PipelineBernardoSilvaC1P1`. Se não estiver, navegue até ela com `cd`.

No **PowerShell**, verifique o Python:

```powershell
python --version
```

### Criar o ambiente virtual (somente na primeira vez)

Se **não existir** uma pasta `.venv` utilizável no projeto:

```powershell
python -m venv .venv
```

Ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

> Se o terminal já começa com `(.venv)`, como na captura de tela, o ambiente virtual **já está ativado**: não precisa executar a ativação novamente. Se a `.venv` estiver em outra pasta, ajuste o caminho do comando.

Se o PowerShell bloquear a execução do script de ativação, uma alternativa temporária, restrita à sessão atual, é:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

No VS Code, selecione o interpretador correto com **Ctrl+Shift+P > Python: Select Interpreter**, escolhendo o Python da `.venv`.

## 4. Instalar as bibliotecas

Primeiro, atualize o `pip`:

```powershell
python -m pip install --upgrade pip
```

Para os scripts cujos imports foram compartilhados, instale pelo menos:

```powershell
python -m pip install numpy pandas pyarrow scipy scikit-learn matplotlib psutil
```

- `numpy`: operações numéricas.
- `pandas`: leitura, organização e gravação de tabelas.
- `pyarrow`: suporte a arquivos Parquet.
- `scipy`: processamento científico, caso utilizado pelos módulos auxiliares.
- `scikit-learn`: classificadores, pré-processamento e métricas.
- `matplotlib`: visualizações e gráficos, caso utilizados.
- `psutil`: gerenciamento e inspeção dos processos ATP em `GerarParquet.py`.

**Importante:** essa lista é uma **base de instalação**, não uma relação exaustiva verificada. Os conteúdos completos de `f_*.py`, `SelecaoAtributos.py` e `Classificadores.py` não foram disponibilizados aqui. Se existir `requirements.txt` no repositório, prefira:

```powershell
python -m pip install -r requirements.txt
```

Se aparecer `ModuleNotFoundError`, confira o nome do módulo, se o arquivo `.py` existe na pasta correta e se a biblioteca externa está instalada no mesmo ambiente virtual. Módulos locais como `f_recortar_janela` **não são pacotes para instalar com `pip`**.

## 5. Executar `ProcessarAtributos.py` (objetivo principal)

### Antes de executar

- Confira se `Parquets/` contém os arquivos `.parquet` das simulações.
- Confira se os módulos `f_recortar_janela.py`, `f_selecionar_sinais_ponto.py`, `f_preparar_sinais.py`, `f_extrair_harmonicas.py`, `f_calcular_parametros.py` e `f_calcular_estatisticas.py` estão junto ao script.
- No início de `ProcessarAtributos.py`, confira as configurações `JANELAS`, `AMOSTRAS_POR_CICLO`, `ORDEM_HARMONICA_MAXIMA` e `REGIOES`.
- O script compartilhado trabalha com **medições da subestação (`SUB`)** e identifica a **fase faltosa pelo nome do arquivo**.
- **A taxa `AMOSTRAS_POR_CICLO` deve ser compatível com os Parquets de entrada.** Não altere de 64 para 16 sem antes reamostrar corretamente os sinais e adaptar a extração harmônica.

### Comando

No terminal aberto na pasta `PipelineBernardoSilvaC1P1`:

```powershell
python .\ProcessarAtributos.py
```

O script:

1. Localiza os arquivos `.parquet` na pasta `Parquets`.
2. Identifica a fase e a barra da falta a partir do nome de cada arquivo.
3. Associa a barra à região definida em `REGIOES`.
4. Recorta os sinais conforme as janelas configuradas.
5. Seleciona os sinais do ponto `SUB` e prepara as representações `soma` e `fase_faltosa`.
6. Extrai componentes harmônicas, calcula parâmetros elétricos e suas estatísticas.
7. Valida os dados e grava os datasets em `DatasetAtributos/`.

No código fornecido, as janelas são `(0, 10)`, `(0, 5)` e `(0, 3)`, relativas ao instante de contato com o solo (**TTOQ**). O script utiliza **apenas o trecho posterior ao TTOQ** nessas três configurações.

### Arquivos gerados

Na versão original compartilhada, os arquivos são:

```text
DatasetAtributos/
├── atributos_0_10.parquet
├── atributos_0_5.parquet
└── atributos_0_3.parquet
```

Na versão posteriormente ajustada para incluir o número de regiões no nome, com `REGIOES` contendo quatro regiões, os nomes passam a ser:

```text
DatasetAtributos/
├── atributos_0_10_4regioes.parquet
├── atributos_0_5_4regioes.parquet
└── atributos_0_3_4regioes.parquet
```

**Verifique qual versão do arquivo `ProcessarAtributos.py` está realmente no projeto antes de procurar os resultados.** O sufixo identifica a regionalização escolhida; não modifica os atributos elétricos extraídos.

## 6. Para gerar novos Parquets a partir do ATP (opcional)

Se os Parquets de entrada ainda não existirem, utilize:

```powershell
python .\GerarParquet.py
```

O `GerarParquet.py` compartilhado:

1. Busca os cartões `.atp` em `TodosCartoesRodadosSemRetirarCargas/CartoesATP/`, filtrando os nomes que contêm `_p1_carregamento1_`.
2. Ignora cartões que já possuem Parquet correspondente em `Parquets/`.
3. Executa o ATP por meio do `runATP.exe`, em diretórios temporários separados.
4. Lê o `.pl4` gerado usando `f_readpl4.py`.
5. Chama `f_processar_pl4.py` para selecionar, recortar e reamostrar os sinais.
6. Salva um Parquet por simulação em `Parquets/`.
7. Valida a quantidade de amostras, colunas e a posição temporal do TTOQ.

Configurações observadas no script fornecido: `NUM_THREADS = 4`, `LIMITE_TESTE = None`, **2.560 linhas esperadas** (40 ciclos × 64 amostras/ciclo), **165 colunas esperadas** e TTOQ no índice 1.280. Se a taxa de amostragem for alterada, esses valores e a função `f_processar_pl4.py` precisam ser adaptados de forma consistente.

> **Cuidado:** gerar os Parquets exige executar simulações ATP e pode demorar. Não execute essa etapa se `Parquets/` já contiver os sinais desejados.

## 7. O que faz cada arquivo principal?

| Arquivo | Função no fluxo |
|---|---|
| `GerarParquet.py` | Executa os cartões ATP, lê os resultados PL4, processa os sinais e gera os arquivos Parquet. |
| `ProcessarAtributos.py` | Lê os Parquets, recorta janelas temporais e extrai atributos elétricos para criar datasets de classificação. |
| `SelecaoAtributos.py` | Etapa de seleção/análise de atributos; **os métodos e os arquivos gerados precisam ser confirmados no próprio código**. |
| `Classificadores.py` | Etapa de treinamento e avaliação dos classificadores; **os modelos e parâmetros da versão local precisam ser confirmados no próprio código**. |
| `Regionalizacoes.py` | Arquivo associado à definição das regionalizações da rede; verificar como é importado pelos scripts de classificação. |
| `AnalisarArquivosCartoesATP.py` | Script de inspeção/análise de cartões ATP; detalhes específicos devem ser verificados no código. |

## 8. O que fazem os módulos auxiliares `f_*.py`?

Esses arquivos são **funções locais reutilizadas pelos scripts principais**. Em geral, não é necessário executá-los individualmente.

| Módulo | Responsabilidade |
|---|---|
| `f_readpl4.py` | Lê os sinais do arquivo `.pl4` gerado pelo ATP. |
| `f_processar_pl4.py` | Processa os sinais do PL4, incluindo seleção, recorte e reamostragem. |
| `f_recortar_janela.py` | Recorta os sinais de acordo com a janela temporal escolhida. |
| `f_selecionar_sinais_ponto.py` | Seleciona os sinais elétricos de um ponto de medição, como `SUB`. |
| `f_selecionar_sinais_medidores.py` | Módulo associado à seleção de sinais de medidores; confirmar a interface no código. |
| `f_preparar_sinais.py` | Prepara as representações dos sinais, incluindo a soma das fases e a fase faltosa. |
| `f_extrair_harmonicas.py` | Extrai componentes harmônicas dos sinais de tensão e corrente. |
| `f_calcular_parametros.py` | Calcula parâmetros elétricos derivados das harmônicas. |
| `f_calcular_estatisticas.py` | Calcula estatísticas dos parâmetros, como média, máximo, mínimo, variância, desvio-padrão e amplitude. |

## 9. Problemas frequentes

**`python` não é reconhecido:** instale/configure o Python ou tente `py --version` e `py -m pip ...`.

**`ModuleNotFoundError: No module named 'pandas'` (ou similar):** confirme que a `.venv` está ativa e instale a biblioteca com `python -m pip install ...`.

**`No module named 'f_recortar_janela'`:** abra o terminal na pasta correta e confira se o arquivo auxiliar existe.

**`Nenhum arquivo .parquet encontrado`:** confira se `Parquets/` existe e contém os arquivos de entrada.

**Erro ao identificar `FaseA` / `FaseB` / `FaseC` ou `barraNN`:** os nomes dos arquivos precisam seguir o padrão esperado pelas expressões regulares de `ProcessarAtributos.py`.

**`A barra ... não pertence a nenhuma região definida`:** verifique se o dicionário `REGIOES` inclui todas as barras presentes nas simulações.

**Erro na quantidade de amostras ou na posição do TTOQ:** confira a compatibilidade entre a taxa de amostragem usada na geração dos Parquets e aquela usada no processamento de atributos.

**`runATP.exe não encontrado`:** verifique o caminho do executável em `GerarParquet.py`; esse erro só importa ao gerar novos Parquets.

**Os arquivos de saída não aparecem:** verifique se a execução terminou sem erros e procure em `DatasetAtributos/`, considerando se os nomes possuem o sufixo `_Nregioes`.

## 10. Checklist

- [ ] Abri `PipelineBernardoSilvaC1P1` no VS Code.
- [ ] Confirmei a versão do Python e selecionei o interpretador da `.venv`.
- [ ] Instalei as dependências necessárias.
- [ ] Confirmei que a pasta `Parquets/` possui os arquivos de entrada.
- [ ] Conferi as configurações de janelas, taxa de amostragem e regionalização.
- [ ] Executei `python .\ProcessarAtributos.py`.
- [ ] Conferi os datasets gerados em `DatasetAtributos/`.

---

**Nota sobre a documentação:** os passos de `GerarParquet.py` e `ProcessarAtributos.py` foram descritos com base nos códigos efetivamente compartilhados. As funções exatas de seleção de atributos, classificadores e alguns módulos auxiliares foram descritas a partir dos nomes e do papel que exercem no fluxo; é necessário consultar os respectivos arquivos para documentar seus parâmetros e resultados com precisão.
