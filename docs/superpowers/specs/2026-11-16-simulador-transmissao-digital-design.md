# Simulador de Transmissão Digital — Design

**Disciplina:** Redes de Computadores (UNISINOS) — Trabalho II (GB)
**Data:** 2026-11-16
**Stack:** Python 3 (NumPy, Matplotlib, PyYAML)
**Prazo:** ~3 dias

## 1. Objetivo

Implementar um simulador de transmissão digital que percorra a cadeia de comunicação
— da mensagem ASCII à recuperação no receptor sob ruído — e conduzir uma avaliação
experimental de **BER (Bit Error Rate)** e **eficiência espectral**, analisando o
impacto de codificação de linha, modulação digital e codificação de canal.

Requisitos oficiais (resumo do enunciado):

- Gerar mensagem ASCII e converter em bits.
- ≥ 2 codificações de linha + explicar diferença entre codificação de linha e de canal.
- ≥ 3 modulações digitais, incluindo uma de ordem superior.
- Canal AWGN com controle de SNR + um segundo cenário (ruído em rajadas).
- Demodular, decodificar, reconstruir e identificar bits incorretos.
- Calcular BER e eficiência espectral nominal (bits/s/Hz).
- Varredura de SNR com ≥ 8 valores; ≥ 100.000 bits por combinação; ≥ 5 repetições aleatórias.
- Curvas BER × SNR com variabilidade; comparar AWGN vs 2º cenário.
- Tabela de compromisso BER × eficiência espectral; teste de consistência.
- Reprodutibilidade: configs, seeds, resultados brutos, scripts.

## 2. Decisões de projeto

| Dimensão | Escolha |
|---|---|
| Linguagem | Python 3 (NumPy, Matplotlib, PyYAML) |
| Modulações | BPSK, QPSK, 16-QAM (com mapeamento Gray) |
| Codificação de linha | Manchester, AMI |
| Codificação de canal | Hamming (7,4) — corrige 1 bit/bloco |
| 2º cenário de canal | Ruído em rajadas (burst) |
| Eixo x das curvas | Eb/N0 (dB) |
| Uso do Hamming | Comparar com/sem (coding gain) |
| Reprodutibilidade | config YAML + seeds fixas + CSV bruto |
| Opcionais incluídos | Curvas teóricas sobrepostas, diagramas de constelação |

### Nota de modelagem: dois pipelines separados

Codificação de linha (Manchester/AMI, banda-base) e modulação passband
(BPSK/QPSK/16-QAM) são níveis distintos da camada física e não se encadeiam
diretamente numa simulação de BER. O simulador os trata como **dois pipelines
paralelos**, cada um medindo BER no seu próprio domínio:

- **Pipeline A — Linha (banda-base):** `bits → Manchester/AMI → canal → detecção → BER`
- **Pipeline B — Modulação (passband):** `ASCII → bits → [Hamming] → BPSK/QPSK/16-QAM → canal → demod → [decodifica Hamming] → bits → BER`

### Distinção codificação de linha × codificação de canal (a documentar no README)

- **Codificação de linha:** adapta o sinal ao meio físico (sincronismo, remoção de
  componente DC, espectro). Não acrescenta redundância para correção. Ex.: Manchester, AMI.
- **Codificação de canal:** acrescenta redundância para **detectar/corrigir erros**
  introduzidos pelo canal. Ex.: Hamming(7,4).

## 3. Arquitetura

```
src/
  message.py         # ASCII <-> bits; localizar bits incorretos
  line_coding.py     # Manchester, AMI (encode/decode)
  channel_coding.py  # Hamming(7,4) (encode/decode via síndrome)
  modulation.py      # BPSK, QPSK, 16-QAM (modulate/demodulate, Gray)
  channel.py         # AWGN, burst noise (calibrados por Eb/N0)
  metrics.py         # BER, eficiência espectral, BER teórico (Q-function)
  experiment.py      # orquestra varredura SNR × repetições; salva CSV
  plots.py           # BER×SNR, constelações, tabela de trade-off
config.yaml          # SNRs, nº de bits, seeds, modulações, repetições
results/             # CSVs brutos (reprodutibilidade)
figures/             # PNGs gerados
tests/               # testes unitários por módulo
README.md            # arquitetura, instalação, parâmetros, reprodução, divisão da dupla
```

Cada módulo tem responsabilidade única, interface clara e é testável isoladamente.

## 4. Componentes (detalhe)

### message.py
- `text_to_bits(s) -> np.ndarray` (8 bits/char, MSB-first), `bits_to_text(bits) -> str`.
- `bit_errors(tx_bits, rx_bits) -> (n_errors, error_positions)`.

### line_coding.py
- **Manchester:** `0 -> [0,1]`, `1 -> [1,0]` (convenção documentada), 2 amostras/bit;
  autossincronizável, sem DC. Decodifica por transição no meio do bit.
- **AMI:** `0 -> 0`, `1 -> ±V` alternando polaridade; controle de DC. Decodifica por limiar.
- Interface: `encode(bits) -> signal`, `decode(signal) -> bits`.

### channel_coding.py
- **Hamming(7,4):** matrizes G (geradora) e H (verificação de paridade). Codifica blocos
  de 4 bits em 7; decodifica por síndrome corrigindo 1 bit/bloco.
- Padding quando o nº de bits não é múltiplo de 4 (registrar bits de padding para remover).
- Interface: `encode(bits) -> coded_bits`, `decode(coded_bits) -> (bits, n_corrected)`.

### modulation.py
- Mapeamento **Gray** para BPSK (1 bit/símbolo), QPSK (2), 16-QAM (4).
- Constelações normalizadas para energia média unitária por símbolo.
- Demodulação por decisão de mínima distância (hard decision).
- Interface: `modulate(bits) -> symbols (complex)`, `demodulate(symbols) -> bits`,
  `bits_per_symbol`, `constellation`.

### channel.py
- **AWGN:** ruído gaussiano complexo calibrado por **Eb/N0**, considerando bits/símbolo
  e taxa de código efetiva (quando Hamming está ativo, R = 4/7).
- **Burst:** AWGN de base + rajadas de ruído forte em janelas aleatórias.
  Parâmetros: probabilidade de início de rajada, duração média, fator de intensidade.
- Interface: `awgn(symbols, ebn0_db, bits_per_symbol, code_rate) -> symbols`,
  `burst(symbols, ebn0_db, ..., rng) -> symbols`. RNG injetado para reprodutibilidade.

### metrics.py
- `ber(tx_bits, rx_bits) -> float`.
- `spectral_efficiency(mod) -> float` (bits/s/Hz nominal = bits/símbolo).
- BER teórico AWGN via Q-function: BPSK/QPSK `Q(sqrt(2·Eb/N0))`;
  16-QAM aproximação padrão para M-QAM.

### experiment.py
- Varre **≥ 8 valores** de Eb/N0 (configuráveis).
- **≥ 100.000 bits** por ponto (configurável).
- **≥ 5 repetições** com seeds derivadas de uma seed-mestre (`np.random.SeedSequence`).
- Agrega média e desvio-padrão da BER; grava CSV bruto por ponto/repetição.

### plots.py
- Curvas BER × Eb/N0 (simulado + teórico sobreposto; barras de erro/variabilidade), eixo y log.
- Diagramas de constelação antes/depois do canal (BPSK/QPSK/16-QAM).
- Tabela BER × eficiência espectral.

## 5. Experimentos

1. **Linha:** Manchester vs AMI sob AWGN e burst → BER × SNR.
2. **Modulação (AWGN):** BPSK/QPSK/16-QAM, com e sem Hamming → BER × Eb/N0 + teórico.
3. **Modulação (burst):** mesmas modulações, com e sem Hamming → ganho de codificação vs rajadas.
4. **Consistência:** canal ideal / Eb/N0 alto → mensagem reconstruída sem erro.
5. **Trade-off:** tabela BER (num Eb/N0 de referência) × eficiência espectral por modulação.
6. **Constelações:** BPSK/QPSK/16-QAM antes e depois do canal.

## 6. Estratégia de testes (TDD)

- `message`: round-trip ASCII↔bits idempotente.
- `line_coding`: encode→decode sem canal recupera bits exatos (Manchester e AMI).
- `channel_coding`: Hamming corrige 1 bit injetado por bloco; round-trip sem erro = identidade.
- `modulation`: modulate→demodulate sem ruído = zero erros (BPSK/QPSK/16-QAM).
- `channel`: potência do ruído calibrada bate com Eb/N0 alvo (tolerância estatística).
- `metrics`: BER simulado aproxima o teórico em Eb/N0 alto dentro de tolerância.

## 7. Reprodutibilidade

- Todos os parâmetros em `config.yaml` (lista de Eb/N0, nº de bits, nº de repetições,
  seed-mestre, parâmetros do burst, modulações ativas).
- Seeds derivadas deterministicamente da seed-mestre.
- Resultados brutos em `results/*.csv`; figuras regeneráveis a partir dos CSVs.
- README documenta como reproduzir todos os gráficos.

## 8. Escopo de entrega

- **Agora:** código completo + testes + README (arquitetura, instalação, parâmetros,
  reprodução). Divisão de responsabilidades da dupla: placeholder no README.
- **Depois:** roteiro de vídeo e gravação (após o trabalho estruturado).
- **Opcionais incluídos:** curvas teóricas sobrepostas + diagramas de constelação.

## 9. Fora de escopo (YAGNI)

- Modulações além de BPSK/QPSK/16-QAM (8-PSK fica como extensão futura trivial).
- Soft-decision decoding, interleaving, filtragem de pulso (raised-cosine).
- Sincronização de portadora/símbolo (assume-se sincronismo ideal).
