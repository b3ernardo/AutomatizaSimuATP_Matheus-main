% 30/11/2022
%Autores: Gabriela Nunes Lopes

%------Esse programa simula automaticamente:
% 1 - FAIs com sinais reais no sistema de 34 barras do IEEE com varios
% medidores
% 2 - Variando o sinal real que parametriza o modelo
% 3 - Variando o carregamento do sistema
% 4 - Com ou sem a presença do gerador distribuido no sistema
% 5 - Variando a penetração do INVERSOR


%-----Devem estar na mesma pasta:
% 1 - Uma pasta como nome "Resultados";
% 2 - O cartão original .atp/ (nesse caso 34barrasGS848completoFINALModeloUmArco.atp)
% 3 - As funções necessárias para rodar o ATP : rodarATP,salvarCartao,runATP.exe,Pl42nat.exe
% 4 - Os dados necessários do sistema: planilhas das linhas das cargas e das FAIs, potencia das cargas
% 5 - As funções de cálculo das cargas e cálculo das FAIs nas barras

clc; clear all;

addpath Functions  % Adicionando a pasta com os arquivos que precisam para rodar
savepath

%% ------- Selecione o que deseja fazer (1 sim e 0 nao)
mudarsinalbase=1; % variar o sinal real que parametriza o modelo
mudarlocfai=1;  % Variar a localização da FAI no sistema
mudarcarregamento=1;  % Variar o carregamento do sistema
mudarpenetracao=1;  % Variar a penetração do gerador
plotar=0;  % se plota as formas de onda

%% ------- Dados iniciaisindice
Fases={'FaseA/';'FaseB/';'FaseC/'};
barrasfasea=[2 3 4 5 6 7 8 10 11 12 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 31 32 33 34];
barrasfaseb=[2 3 4 5 6 7 8 9 11 13 14 16 17 18 19 20 21 22 24 25 26 27 28 29 30 31 32 33 34];
barrasfasec=[2 3 4 5 6 7 8 11 14 16 17 19 20 21 22 24 25 26 27 28 29 31 32 33 34];
sinaistestados=[3 9 13 20 25 26 27 28 31 33];
ruptt=2.5; % ruptura do condutor
toucht=3.78; % toque do condutor ao solo
randomizar=1; % se é pro modelo ser aleatório
carregamento=[1 0.3]; % definir os carregamento que serão simulados
Pgd=[0.995 0.665 0.335]; % definir os niveis de penetração da GD
Qgd=[0.0995 0.0665 0.0335]; % O nivel de penetração de potência reativa deve ser 10% da ativa


% Dados necessários de informações do sistema
Linhascd=xlsread('linhascargasdistribuidas.xlsx'); %Linhas do cartao com as cargas distribuidas
Linhascp=xlsread('linhascargaspontuais.xlsx'); %Linhas do cartao com as cargas pontuais
LinhasBarras=xlsread('LinhasHIFReal.xlsx');
LinhasModeloFAI=xlsread('LinhasParametrosHIF.xlsx'); % linhas onde estao os parametros do modelo de fai
linhatsimu=9;  % linha do tempo de simulacao
linhamudasolo=308;  % LINHA DO SOLO
ParametrosIniciais={'24900.';'60.';'5.E3';'3.';'5.28';'10.';'10.';'10.';'0.';...
    '10.';'10.';'10.';'10.';'10.';'10.';'10.';'10.';'10.';'10.';'1.';};

%% ---- Primeira parte: Rodando o cartão do ATP:

%Carrega cartão original
cardNameatporiginal = '34barrasGS848_FAImodelo_ComMedidores.atp'; %(teste2 eu mudei a resist) %Cartao onde a simu do ATP está
% rodando pro sem GD
% cardNameatporiginal = '34barrasGS848_FAIreal_Sistema2022_paraSGD.atp';
cardNameatpalterado = 'atpalterado.atp';    % Cria cartão de cópia para poder modificá-lo;
cardNamepl4 = 'atpalterado.pl4';
start = datetime;
delay = 3; % tempo de espera para comecar a converter o pl4
%fprintf('Leitura do cartão \n');
fid = fopen(cardNameatporiginal,'rt+');
i = 1;
tline = fgetl(fid);
A{i} = tline;
while ischar(tline)
    i = i+1;
    tline = fgetl(fid);
    A{i} = tline;
end
fclose(fid);


%% Definindo a quantidade de simulacoes de acordo com as modificacoes requeridas


if mudarcarregamento==0; fcar=1; else fcar=length(carregamento); end
if mudarpenetracao==0; fpen=1; else fpen=length(Pgd); end


%% Segunda parte: Modificação nas simulações
%A= cartão original
%B= Cartao de cópia que especifica com ou sem a GD
%C= cartão de cópia com a penetração da GD
%D= Cartao de copia com o novo carregamento
%E= Cartao de cópia para nova parametrização da FAI
%F= Cartao de copia para novo local da fai quantas simus necessarias


%% B - Simulando com ou sem GD
for fs=1:length(Fases)

    fase=char(Fases(fs));
    if fs==1
        barrassimu=barrasfasea;
    elseif fs==2
        barrassimu=barrasfaseb;
    else
        barrassimu=barrasfasec;
    end


    if mudarlocfai==0; flocfai=1; else flocfai=length(barrassimu); end


    for cgd=1%:2
        B=A; % Fazendo a primeira copia do cartao original

        if cgd==2 % se quiser simular o sistema sem a GD (o cartao original é COM gd)
            for lchavesgd=1788:1790   %1195 1196 1197 linhas onde a chave da GD estão
                replace = strrep(B{1,lchavesgd},'-1.','10.'); % a chave fica aberta
                B{1,lchavesgd} = replace;
            end
            %adiciona o cap 2 de volta
            for lchavescap=1665:1667   %1195 1196 1197 linhas onde a chave da GD estão
                replace = strrep(B{1,lchavescap},'1.E3      1.E3      1.E3',' -1.      1.E3      1.E3'); % a chave fica aberta
                B{1,lchavescap} = replace;
            end
        end

        %% C -  Modificando a penetração da GD


        for pn=1:fpen % para todos os niveis de penetração
            C=B; % Faz a cópia para modificar o carregamento

            if pn>1
                %muda potencia ativa
                replace = strrep(C{1,12},'0.995',num2str(Pgd(pn)));
                C{1,12} = replace;
                %muda potencia reativa
                replace = strrep(C{1,13},'0.0995',num2str(Qgd(pn)));
                C{1,13} = replace;
            end

            %% D - Mudando o carregamento
            for car=1:fcar % para todos os carregamentos possiveis
                D=C; % faz a cópia do cartão para mudar o carregamento


                if car>1 % só muda o carregamento se for outro além do nominal
                    % calcula a impedancia para o novo carregamento
                    [D] = mudacarregamentov2(D,carregamento(car),Linhascd,Linhascp);
                end


                %% E: Inicio simulacoes FAIs

                %Como a Barra 1 é a subestação e o modelo já está na barra 2:
                for pb= 1:length(barrassimu)  %Para todas as barras
                    proxbarra=barrassimu(pb);
                    barra=2;  % A barra em que já está é a 2
                    noI=1664; % linhas onde estao inicialmente o inicio do moldeo
                    noF=1537; % e o fim do modelo
                    linhach=0000; % linha entre o modelo e a chave do modelo da fai
                    CaminhoDados='G:\Meu Drive\DOUTORADO\simulacoesATP\HIFmodeloLuiz\FAImodelo_comtqueda\arquivospararodar\sinaisreais';
                    files=dir(strcat(CaminhoDados,'\*.mat'));  %leitura de todos os dados

                    if mudarsinalbase==0; fsb=1; else fsb=length(files); end

                    for solos =1:length(sinaistestados)  % para todos os sinais base
                        E = D;
                        sb=sinaistestados(solos);
                        nomeSinal=num2str(sb);
                        [Parametros]=dadosparametrizacao(CaminhoDados,files,nomeSinal,ParametrosIniciais);
                        [E]=parametrizamodelo(E,Parametros,ParametrosIniciais,LinhasModeloFAI);
                        F=E;
                        [F,str_proxbarra] = faino34barras_HIFmodelo2022(E,F,barra,proxbarra,LinhasBarras,fs,noI,noF,solos,linhatsimu,linhamudasolo,linhach);

                        %% ----------- salvando o novo cartão do ATP
                        pause(1)
                        salvacartaonapasta(F,cardNameatpalterado,str_proxbarra,fase,cgd,pn,car,solos);

                        pause(2)

                    end % end dos solos/sinais de FAI
                end % end das localizações da FAI
            end % end carregamentos
        end % end niveis de penetração
    end % end com Gd e sem Gd
    clear barrassimu
end % fases