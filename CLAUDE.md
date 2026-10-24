# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## O que é

Evolução de robôs construíveis: MAP-Elites (pyribs) sobre corpos, CMA-ES sobre um CPG em malha aberta para cada corpo, simulação no MuJoCo. Os corpos usam peças reais (servo Feetech STS3215 7,4 V, links impressos, ESP32 + LiPo) e qualquer robô do arquivo pode ser exportado como STL, lista de materiais, guia de montagem e JSON de controle. Projeto pessoal, hobby e portfólio. Navegação e firmware ainda não existem (o firmware espera o hardware).

Código, comentários e README em inglês; `docs/` e este arquivo em português.

## Comandos

Rodar da raiz, com o venv ativo (`venv/`, Python 3.10).

```bash
pip install -e ".[video,cad]"                 # gpu: venv separado (venv-gpu) com ".[gpu]"
pytest tests -q                               # ~65 testes, ~20 s; os de GPU se pulam sem jax
python experiments/train.py --terrains terrain --batches 30 --workers 3
python experiments/replay.py --list           # tipos de robô no arquivo
python experiments/replay.py --type 6legs-0wheels-12servos --speed 0.5
python experiments/validate.py --type <tipo>  # PASS se perde < 30% em física variada
python experiments/export.py --type <tipo>    # exports/<tipo>/
python experiments/plot_archive.py
```

A máquina do dono é um notebook de 4 núcleos: comandos pesados com `nice -n 19`, no máximo 1 worker em testes e treinos curtos de verificação.

## Arquitetura

`darwinbot/`:

- `morphology.py`: genoma do corpo em [0, 1], até 3 pares de pernas espelhadas; `descriptor()` dá os eixos do mapa (pares, pares com roda, nº de servos).
- `controller.py`: genoma do CPG de tamanho fixo (genes de pares ou rodas ausentes são ignorados, o que permite herdar o controlador do pai). Amplitude cortada pela velocidade do servo.
- `builder.py`: morfologia + terreno -> MJCF e o `ActuatorLayout`. **A ordem dos atuadores no layout tem de bater com a ordem no modelo**; o CPG depende disso.
- `sim.py`: `Episode` compila uma vez e roda muitos controladores; controle a 50 Hz (alvo segurado entre atualizações, como no ESP32); corpo com peças sobrepostas no spawn não é construível.
- `sim_gpu.py`: mesma coisa em lote com MJX. **Qualquer mudança de física ou controle em `sim.py` precisa ser espelhada aqui**; `tests/test_sim_gpu.py` compara os dois (roda só no venv-gpu, ~3 min na 940MX local).
- `randomize.py`: variações de física (atrito, massa, torque, rigidez, folga, atraso) para o treino e o `validate.py`.
- `inner.py`: CMA-ES por corpo; `evolve.py`: arquivo pyribs, laço externo, checkpoints.
- `parts.py`, `control_file.py`, `export.py`: peças build123d, JSON do firmware (`docs/formato-controle.md`), pacote de exportação.

## Convenções e armadilhas

- Mudou o genoma ou a física de forma incompatível: incrementar `Archive.body_model` em `evolve.py`; arquivos antigos passam a ser recusados com mensagem clara.
- Valores de hardware vêm de `docs/hardware-dados.md`. As medidas do servo em `parts.py` são provisórias ("measure a real STS3215 and update") até alguém medir um servo real.
- `.scratch/` é área de rascunho fora do git; `checkpoints/`, `exports/` e `*.mp4` também ficam fora.
- Commits sem menção ao Claude.
