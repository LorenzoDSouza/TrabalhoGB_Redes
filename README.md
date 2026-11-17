# Simulador de Transmissão Digital

Trabalho GB — Redes de Computadores (UNISINOS)

## Objetivo

Simular, de forma reprodutível, uma cadeia completa de transmissão digital —
de uma mensagem em texto ASCII até sua recuperação após passar por um canal
ruidoso — e medir o desempenho (BER × Eb/N0) de diferentes esquemas de
codificação e modulação. O simulador cobre dois pipelines paralelos:

- **Banda-base (codificação de linha):** Manchester e AMI.
- **Banda-passante (modulação):** BPSK, QPSK e 16-QAM, com codificação de
  canal opcional Hamming(7,4).

A partir dos resultados são gerados gráficos de BER × Eb/N0 (com curvas
teóricas de referência), diagramas de constelação e uma tabela de trade-off
entre eficiência espectral e taxa de erro.

## Arquitetura do simulador

Dois pipelines independentes compartilham os mesmos módulos de canal e métricas:

```
  Pipeline de modulação (banda-passante)
  ───────────────────────────────────────────────────────────────────────
  texto ──▶ bits ──▶ [Hamming(7,4)?] ──▶ modula ──▶ canal ──▶ demodula
                                              (AWGN / burst)      │
                     bits recuperados ◀── [Hamming decode?] ◀─────┘
                                              │
                                              ▼
                                      BER / bits errados

  Pipeline de codificação de linha (banda-base)
  ───────────────────────────────────────────────────────────────────────
  texto ──▶ bits ──▶ codifica linha ──▶ canal (ruído real) ──▶ decodifica
                     (Manchester/AMI)   (AWGN / burst)             │
                                   bits recuperados ◀──────────────┘
                                              │
                                              ▼
                                      BER / bits errados
```

| Módulo                  | Responsabilidade                                                              |
| ----------------------- | ----------------------------------------------------------------------------- |
| `src/message.py`        | Conversão ASCII ↔ bits (8 bits/char, MSB-first) e localização de bits errados |
| `src/line_coding.py`    | Codificação/decodificação de linha: Manchester e AMI                          |
| `src/channel_coding.py` | Codificação de canal Hamming(7,4) com correção de 1 bit/bloco (R = 4/7)       |
| `src/modulation.py`     | Modulações BPSK/QPSK/16-QAM com mapeamento Gray e energia média unitária      |
| `src/channel.py`        | Canais AWGN e burst, calibrados por Eb/N0                                     |
| `src/metrics.py`        | BER, eficiência espectral e curvas teóricas de BER (AWGN)                     |
| `src/pipeline.py`       | Orquestração ponta-a-ponta dos dois pipelines                                 |
| `src/experiment.py`     | Leitura de `config.yaml` e varredura de BER × Eb/N0 com repetições/seeds      |
| `src/plots.py`          | Geração de curvas de BER, constelações e tabela de trade-off                  |
| `main.py`               | CLI que roda a varredura completa, gera figuras e o teste de consistência     |

## Instalação

```bash
python3 -m pip install -r requirements.txt
```

Dependências: NumPy, Matplotlib, PyYAML e pytest. Não há dependência de SciPy —
a função Q (erfc) é calculada com a biblioteca padrão.

## Execução

```bash
python3 main.py
```

Isso executa:

1. A varredura completa definida em `config.yaml`, escrevendo os resultados
   brutos em `results/ber_results.csv`.
2. A geração de todas as figuras em `figures/` (curvas de BER por canal e por
   uso de Hamming, curvas de codificação de linha, e constelações).
3. A impressão da tabela de trade-off (eficiência espectral × BER).
4. O teste de consistência: reconstrução da mensagem sem erros em Eb/N0 alto.

Para rodar os testes automatizados:

```bash
python3 -m pytest
```

## Parâmetros (`config.yaml`)

| Chave                  | Descrição                                                                 |
| ---------------------- | ------------------------------------------------------------------------- |
| `master_seed`          | Semente-mestre; todas as sementes de repetição derivam dela              |
| `n_bits`               | Número de bits por ponto da curva (≥ 100.000)                             |
| `n_reps`               | Número de repetições por condição (≥ 5), usadas para média e desvio      |
| `ebn0_db`              | Lista de valores de Eb/N0 em dB (≥ 8 valores)                            |
| `message`              | Mensagem usada no teste de consistência                                   |
| `modulations`          | Modulações avaliadas: `BPSK`, `QPSK`, `16-QAM`                            |
| `line_codes`           | Códigos de linha avaliados: `manchester`, `ami`                           |
| `channels`             | Canais avaliados: `awgn`, `burst`                                         |
| `use_hamming_options`  | Comparação com e sem Hamming(7,4): `[false, true]`                        |
| `burst.burst_prob`     | Probabilidade de iniciar uma rajada de ruído                              |
| `burst.burst_len`      | Comprimento (em símbolos) de cada rajada                                  |
| `burst.burst_factor`   | Fator de amplificação do ruído durante a rajada                           |

## Reprodutibilidade

- **Todos** os parâmetros do experimento vivem em `config.yaml`.
- A aleatoriedade é controlada por uma `master_seed`. A partir dela,
  `numpy.random.SeedSequence` gera uma semente por repetição, reutilizada entre
  as condições para comparação justa. Nenhuma função usa o gerador global do
  NumPy — um `numpy.random.Generator` é sempre injetado explicitamente.
- Os resultados brutos (`results/ber_results.csv`) são versionados no git.
- As figuras (`figures/`) **não** são versionadas, pois são reproduzíveis a
  partir dos scripts + CSV. Rode `python3 main.py` para regenerá-las.

## Codificação de linha × codificação de canal

Embora ambas tenham a palavra "codificação", elas atuam em camadas e com
objetivos diferentes:

- **Codificação de linha** (ex.: Manchester, AMI) transforma a sequência de
  bits em um **sinal físico** adequado ao meio de transmissão em banda-base. Os
  objetivos são elétricos/de sincronização: garantir transições suficientes
  para recuperação de clock (Manchester), eliminar componente DC (AMI),
  controlar o espectro do sinal. **Não adiciona redundância para correção de
  erros** — a taxa de informação por símbolo não aumenta a robustez contra
  ruído; ela molda o sinal.

- **Codificação de canal** (ex.: Hamming(7,4)) adiciona **redundância
  controlada** aos dados (bits de paridade) para **detectar e corrigir erros**
  introduzidos pelo canal. O Hamming(7,4) usa 3 bits de paridade para cada 4
  bits de dados (taxa R = 4/7) e corrige 1 erro por bloco de 7 bits. O custo é
  uma redução da taxa útil de informação em troca de menor BER efetiva.

Em resumo: codificação de linha é sobre **como representar o sinal no meio**;
codificação de canal é sobre **proteger a informação contra erros**. No
simulador, o pipeline de linha opera em banda-base com ruído aplicado às
amostras do sinal, enquanto o pipeline de modulação pode (opcionalmente)
aplicar Hamming(7,4) antes da modulação, permitindo comparar diretamente o
ganho de codificação de canal.

## Resultados e discussão

A tabela de trade-off (AWGN, sem Hamming) resume a tensão fundamental entre
eficiência espectral e robustez: modulações de ordem mais alta transmitem mais
bits por símbolo (maior eficiência espectral) ao custo de uma BER maior para um
mesmo Eb/N0. BPSK e QPSK têm desempenho de BER essencialmente idêntico em
função de Eb/N0, mas QPSK dobra a eficiência espectral; 16-QAM dobra novamente
a eficiência, porém exige um Eb/N0 bem maior para atingir a mesma BER.

<!-- preencher após rodar os experimentos: colar as figuras geradas em figures/
     e discutir as curvas BER × Eb/N0 (AWGN vs burst, com vs sem Hamming) e a
     tabela de trade-off eficiência espectral × BER -->

## Divisão de responsabilidades da dupla

<!-- Integrante A: ... / Integrante B: ... -->

## Vídeo

<!-- adicionar o link do vídeo de apresentação -->
