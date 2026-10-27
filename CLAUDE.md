# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é

Evolução de robôs construíveis: MAP-Elites (pyribs, arquivo CVT por comportamento e custo) sobre corpos em árvore de módulos, CMA-ES sobre um CPG com dois harmônicos e retorno de carga (Tegotae) para cada corpo, simulação no MuJoCo. Os corpos usam peças reais (servo Feetech STS3215 7,4 V, links impressos, ESP32 + LiPo) e qualquer robô do arquivo pode ser exportado como STL, lista de materiais, guia de montagem e JSON de controle. Projeto pessoal, hobby e portfólio. Navegação ainda não existe. O núcleo do firmware existe (C++ portátil em `firmware/core`, com teste de paridade contra o Python); falta só a camada do ESP32, que espera o hardware.

Código, comentários e README em inglês; `docs/` e este arquivo em português. O guia para iniciantes existe nas duas línguas (`docs/guia-evolucao.md` e `docs/evolution-guide.md`); mudou um, mude o outro. Figuras do guia: `tools/make_guide_figures.py`.

## Comandos

Rodar da raiz, com o venv ativo (`venv/`, Python 3.10).

```bash
pip install -e ".[video,cad]"                 # gpu: venv separado (venv-gpu) com ".[gpu]"
pytest tests -q                               # testes, ~1 min; os de GPU se pulam sem jax
python experiments/train.py --terrains terrain --batches 30 --workers 3
python experiments/train.py --parent-selection uniform   # padrão: novelty
python experiments/replay.py --list           # tipos de robô no arquivo, com duty, altura, servos e custo de transporte (cot)
python experiments/replay.py --type 12servos-high-trot --speed 0.5
python experiments/validate.py --type <tipo>  # PASS se perde < 30% em física variada
python experiments/export.py --type <tipo>    # exports/<tipo>/
python experiments/plot_archive.py
```

A máquina do dono é um notebook de 4 núcleos: comandos pesados com `nice -n 19`, no máximo 1 worker em testes e treinos curtos de verificação.

## Arquitetura

`darwinbot/`:

- `morphology.py`: corpo em árvore de módulos (`core`, `joint`, `link`, `wheel`, `foot` e os passivos sem servo `spring`, `tail`, `skid`, `spike`, `caster`), até `MAX_NODES = 20`, `MAX_SERVOS = 16`, profundidade 5. Genoma de tamanho fixo, um bloco de 13 genes por vaga de nó, todos medidas contínuas: comprimento do link 40 a 200 mm, seção 12 a 30 x 3 a 10 mm, cotovelo (distância 25 a 80 mm, ângulo 30 a 150°), rotação qualquer, posição e guinada do encaixe na borda do tronco, raio do pé 8 a 30 mm. `violations()` aplica as regras de impressão (tensão de flexão do link sob o torque de travamento <= 25 MPa, peças <= 220 mm); corpo que quebra uma regra não é construível e nunca é simulado.  `decode()` aceita qualquer vetor e sempre devolve um corpo válido. A vaga é o id estável do nó. Mutação na árvore (parâmetro contínuo, com um salto grande a cada cinco passos, esticar ou encolher um membro inteiro, tipo, inclusive virar módulo passivo, adicionar, remover, duplicar espelhado, copiar subárvore). Cruzamento por enxerto (`graft`): um membro inteiro de outra elite vai para uma vaga livre, em 20% dos filhos (`GRAFT_P` em `evolve.py`). A semente do treino é sorteada a cada execução e impressa no log.
- `controller.py`: genoma do controlador de tamanho fixo, indexado pela vaga do nó (genes de vagas ausentes são ignorados, o que permite herdar o controlador do pai). Por junta: `amp1, phase1, offset, amp2, phase2, sigma`; por roda: velocidade. Oscilador de fase por junta com retorno de carga (Tegotae): `dtheta = omega - sigma * N * cos(theta)`, `N` = carga do servo sobre o torque máximo. Amplitude cortada pela velocidade do servo.
- `builder.py`: árvore + terreno -> MJCF e o `ActuatorLayout`. **A ordem dos atuadores no layout tem de bater com a ordem no modelo** (profundidade, filhos por vaga); o controlador depende disso. A massa do link sai da geometria (PLA); mola, cauda, patim, espeto e roda livre entram como juntas passivas ou geoms com atrito próprio, sem atuador.
- `sim.py`: `Episode` compila uma vez e roda muitos controladores; mede os descritores de comportamento (duty, altura); `theta` é estado da simulação; controle a 50 Hz (alvo segurado entre atualizações, como no ESP32); corpo com peças sobrepostas no spawn não é construível.
- `sim_gpu.py`: mesma coisa em lote com MJX. **Qualquer mudança de física ou controle em `sim.py` precisa ser espelhada aqui**; `tests/test_sim_gpu.py` compara os dois (roda só no venv-gpu, ~3 min na 940MX local).
- Tarefas: o arquivo guarda `task` (`forward` ou `turn`, `fitness.TASKS`) e `payload` (kg extra no tronco, vai para `build_mjcf`/`Episode`/`BatchEpisode`); `replay`, `validate` e `export` leem os dois do arquivo. `Metrics.yaw` é a variação acumulada do rumo do tronco (calculada igual em `sim.py` e `sim_gpu.py`). O terreno `step` é um degrau de 3 cm a 0,50 m do spawn. Terrenos em `terrain.py` (`TERRAINS`, conjuntos em `PRESETS`); `rough` e `rubble` são aleatórios e o treino sorteia um novo para cada corpo (`get_terrains(spec, seed)`), `replay`/`validate` usam o fixo. `Terrain.spawn_z` é a altura do chão no spawn (`downhill` começa num planalto). Genoma e descritores não mudam com a tarefa, então `BODY_MODEL` fica.
- `randomize.py`: variações de física (atrito, massa, torque, rigidez, folga, atraso) para o treino e o `validate.py`.
- `experiments/lineage.py`: genealogia gravada em `Archive.lineage` (pai de cada elite, inclusive os substituídos).
- `inner.py`: CMA-ES por corpo; `evolve.py`: arquivo CVT do pyribs (~500 células; descritores duty, altura do tronco sobre o alcance, nº de pernas (`Morphology.n_legs`) e custo de transporte em escala log, medidos no terreno plano nominal), seleção de pais por novidade (`--parent-selection novelty|uniform`), nomes de tipo (`8servos-mid-trot`: servos, faixa de altura `low/mid/high`, marcha `crawl/walk/trot/roll`), laço externo, checkpoints. Corpos com mais de 4 juntas recebem mais avaliações do CMA-ES (`scaled_budget`, até 3x).
- `firmware/core`: o mesmo CPG em C++17 sem dependências (`darwin_control`), que divide a carga por `load_full_scale` e dá a volta na fase a cada tick; `tests/test_firmware_parity.py` compila e compara alvos e velocidades de roda com `controller.py`.
- `parts.py`, `control_file.py`, `export.py`: peças build123d com as medidas de cada robô, JSON do firmware (formato 2, com `load_full_scale`, `docs/formato-controle.md`), pacote de exportação que percorre a árvore: um STL por tamanho distinto (`link_{L}x{W}x{T}`, `elbow_{D}mm_{A}deg_{W}x{T}`, `foot_{R}`, `spring_{W}x{B}`, `tail_{L}`, `skid`, `spike`, `caster_wheel_{R}`, `caster_fork`), tronco em placas do tamanho da mesa (`torso_tile_{i}`, ou `torso_plate` se couber) com placas de emenda (`splice_plate_{span}`), e uma linha de montagem por servo na ordem do layout.

## Planejado

- Tarefa `getup` (levantar-se): o robô nasce de cabeça para baixo, a queda não encerra o episódio e a pontuação mede se ele termina em pé. Mexe em `sim.py` e `sim_gpu.py` (orientação do spawn e detecção de queda) e em `fitness.py`.
- Direção por comando (próxima mudança, depois do treino do eixo de pernas): entrada `c` de -1 a +1 que modula amplitude e fase de cada junta e a velocidade das rodas; três episódios por avaliação (reto, esquerda, direita) e nota que combina andar reto e virar quando mandado.
- Forma livre das peças impressas (camada 3): tronco com contorno livre ou segmentado, links curvos, pés variados, encaixes em qualquer face.
- Terrenos adversários (ideia do "parasita" de Hillis e do POET): guardar os terrenos sorteados e as variações de física em que as elites mais falham e avaliar todos também neles; um "hall da fama" desses casos deixa o `validate.py` mais rigoroso. Vem depois da forma livre e antes do `getup`.
- Sensor de equilíbrio: um IMU no ESP32 (MPU-6050, por exemplo) com a inclinação do tronco entrando no controlador, para cada perna corrigir o tombo. Hoje o controlador só sente a carga de cada servo, então a evolução prefere corpos estáveis por si (largos, baixos, com muitos apoios ou rodas); um bípede de pé precisa disso. Mexe em `controller.py`, `sim.py`, `sim_gpu.py`, firmware e `BODY_MODEL`.
- Predador e presa, bem mais adiante: perseguidor programado e depois um segundo arquivo que evolui junto. Depende de sensores no robô.

## Convenções e armadilhas

- Mudou o genoma ou a física de forma incompatível: incrementar `BODY_MODEL` em `evolve.py`; arquivos antigos passam a ser recusados com mensagem clara.
- Valores de hardware vêm de `docs/hardware-dados.md`. As medidas do servo em `parts.py` são provisórias ("measure a real STS3215 and update") até alguém medir um servo real.
- `.scratch/` é área de rascunho fora do git; `checkpoints/`, `exports/` e `*.mp4` também ficam fora.
- Commits sem menção ao Claude.
