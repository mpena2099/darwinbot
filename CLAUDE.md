# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é

Evolução de robôs construíveis: MAP-Elites (pyribs, arquivo CVT por comportamento e custo) sobre corpos em árvore de módulos, CMA-ES sobre um CPG com dois harmônicos e retorno de carga (Tegotae) para cada corpo, simulação no MuJoCo. Os corpos usam peças reais (servo Feetech STS3215 7,4 V, links impressos, ESP32 + LiPo) e qualquer robô do arquivo pode ser exportado como STL, lista de materiais, guia de montagem e JSON de controle. Projeto pessoal, hobby e portfólio. Navegação e firmware ainda não existem (o firmware espera o hardware).

Código, comentários e README em inglês; `docs/` e este arquivo em português.

## Comandos

Rodar da raiz, com o venv ativo (`venv/`, Python 3.10).

```bash
pip install -e ".[video,cad]"                 # gpu: venv separado (venv-gpu) com ".[gpu]"
pytest tests -q                               # testes, ~1 min; os de GPU se pulam sem jax
python experiments/train.py --terrains terrain --batches 30 --workers 3
python experiments/train.py --parent-selection uniform   # padrão: novelty
python experiments/replay.py --list           # tipos de robô no arquivo, com duty, altura e servos
python experiments/replay.py --type 12servos-high-trot --speed 0.5
python experiments/validate.py --type <tipo>  # PASS se perde < 30% em física variada
python experiments/export.py --type <tipo>    # exports/<tipo>/
python experiments/plot_archive.py
```

A máquina do dono é um notebook de 4 núcleos: comandos pesados com `nice -n 19`, no máximo 1 worker em testes e treinos curtos de verificação.

## Arquitetura

`darwinbot/`:

- `morphology.py`: corpo em árvore de módulos (`core`, `joint`, `link`, `wheel`, `foot`), até `MAX_NODES = 20`, `MAX_SERVOS = 16`, profundidade 5. Genoma de tamanho fixo, um bloco por vaga de nó; `decode()` aceita qualquer vetor e sempre devolve um corpo válido. A vaga é o id estável do nó. Mutação na árvore (parâmetro, tipo, adicionar, remover, duplicar espelhado, copiar subárvore).
- `controller.py`: genoma do controlador de tamanho fixo, indexado pela vaga do nó (genes de vagas ausentes são ignorados, o que permite herdar o controlador do pai). Por junta: `amp1, phase1, offset, amp2, phase2, sigma`; por roda: velocidade. Oscilador de fase por junta com retorno de carga (Tegotae): `dtheta = omega - sigma * N * cos(theta)`, `N` = carga do servo sobre o torque máximo. Amplitude cortada pela velocidade do servo.
- `builder.py`: árvore + terreno -> MJCF e o `ActuatorLayout`. **A ordem dos atuadores no layout tem de bater com a ordem no modelo** (profundidade, filhos por vaga); o controlador depende disso.
- `sim.py`: `Episode` compila uma vez e roda muitos controladores; mede os descritores de comportamento (duty, altura); `theta` é estado da simulação; controle a 50 Hz (alvo segurado entre atualizações, como no ESP32); corpo com peças sobrepostas no spawn não é construível.
- `sim_gpu.py`: mesma coisa em lote com MJX. **Qualquer mudança de física ou controle em `sim.py` precisa ser espelhada aqui**; `tests/test_sim_gpu.py` compara os dois (roda só no venv-gpu, ~3 min na 940MX local).
- `randomize.py`: variações de física (atrito, massa, torque, rigidez, folga, atraso) para o treino e o `validate.py`.
- `inner.py`: CMA-ES por corpo; `evolve.py`: arquivo CVT do pyribs (~300 células; descritores duty, altura do tronco sobre o alcance e nº de servos, medidos no terreno plano nominal), seleção de pais por novidade (`--parent-selection novelty|uniform`), nomes de tipo (`8servos-mid-trot`: servos, faixa de altura `low/mid/high`, marcha `crawl/walk/trot/roll`), laço externo, checkpoints.
- `parts.py`, `control_file.py`, `export.py`: peças build123d, JSON do firmware (formato 2, `docs/formato-controle.md`), pacote de exportação que percorre a árvore (um STL por peça distinta, uma linha de montagem por servo na ordem do layout).

## Convenções e armadilhas

- Mudou o genoma ou a física de forma incompatível: incrementar `BODY_MODEL` em `evolve.py`; arquivos antigos passam a ser recusados com mensagem clara.
- Valores de hardware vêm de `docs/hardware-dados.md`. As medidas do servo em `parts.py` são provisórias ("measure a real STS3215 and update") até alguém medir um servo real.
- `.scratch/` é área de rascunho fora do git; `checkpoints/`, `exports/` e `*.mp4` também ficam fora.
- Commits sem menção ao Claude.
