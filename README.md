# DarwinBot

**Evolving robots designed to be built.**

[![tests](https://github.com/mpena2099/darwinbot/actions/workflows/tests.yml/badge.svg)](https://github.com/mpena2099/darwinbot/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Quality-diversity evolution of robot bodies and locomotion controllers, built
from real parts so the result can be printed and assembled.

![Four evolved robots moving: two rollers, a trotter and a walker](docs/img/gait.gif)

Every robot is a tree of real modules (Karl Sims and Revolve2 style) driven by
Feetech STS3215 servos. Bodies are evolved with a MAP-Elites archive laid out by
behaviour and cost, while a central pattern generator with load feedback
(Tegotae) is tuned with CMA-ES for each candidate body. Any robot in the archive
can be exported as printable parts, a bill of materials and a control file for an
ESP32.

> **Status:** the robots are evolved and validated in simulation only. No robot
> has been printed or assembled yet, the parts have never been fitted together,
> and the servo measures they are built around are placeholders until a real
> STS3215 is measured. Treat an export as a design to check, not a kit known to
> work. Building the first one is the next milestone
> ([roadmap](https://mpena2099.github.io/darwinbot/roadmap/)).

## What evolution came up with

| Body plans found in one run | Printable parts of the best robot |
|---|---|
| ![Evolved body plans](docs/img/robots.png) | ![Printable parts of the best robot](docs/img/parts.png) |
| Wheels, legs and hybrids, each the best of its region of the map. | Every part fits a 220 mm print bed, and links pass a simple bending check at the servo's stall torque. |

Trained on four environments: flat floor, bumps, low steps and rubble
([all environments](https://mpena2099.github.io/darwinbot/environments/)).

[Full results →](https://mpena2099.github.io/darwinbot/results/) (numbers, the
archive map and training progress)

## In three commands

```bash
python -m venv venv && source venv/bin/activate && pip install -e .   # install
python experiments/train.py                                           # evolve
python experiments/replay.py --list                                   # see the robot types, then:
python experiments/replay.py --type 8servos-mid-trot                  # watch one
```

## Documentation

Full documentation: <https://mpena2099.github.io/darwinbot/> (online once the
repository is public). To preview it locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

- [Gallery](docs/site/index.html): interactive map of the archive and a 3D replay of each robot
- Beginner guide to evolutionary algorithms: [English](docs/evolution-guide.md) · [Português](docs/guia-evolucao.md)
- [Control file format](docs/formato-controle.md) and [hardware data](docs/hardware-dados.md) (Portuguese)
- [Contributing](CONTRIBUTING.md)

## Resumo em português

Quem está começando com algoritmos evolutivos pode ler antes o
[guia em português](docs/guia-evolucao.md).

O DarwinBot inventa robôs e ensina cada um a andar, numa simulação física
(MuJoCo). A evolução decide o corpo, uma árvore de módulos: quantas pernas, de que
tamanho (medidas contínuas, em milímetros), em que direção cada junta gira, onde
cada peça se prende no tronco, e se a perna termina numa roda. Há também módulos
passivos sem servo: mola de TPU, cauda com contrapeso, patim, espeto e roda
livre. Só entram corpos que a impressora e o servo aguentam: peças de até
220 mm, links que não dobram sob o torque do servo e tronco cortado em placas
emendadas. Para cada
corpo, um otimizador (CMA-ES) procura o ritmo de movimento das juntas, com dois
harmônicos e um retorno da carga dos servos, que leva o robô mais longe. Em vez de
um único vencedor, o programa guarda o melhor robô de cada comportamento (MAP-Elites
por tempo de contato com o chão, altura do tronco e número de servos), formando um
catálogo de soluções diferentes. Os robôs são montados com peças reais (servos
STS3215, peças impressas em 3D, ESP32 e bateria LiPo), treinados com física
propositalmente variada para resistir às diferenças entre simulação e mundo real,
e qualquer um deles pode ser exportado como peças STL, lista de materiais, guia de
montagem e arquivo de controle. Código e documentação técnica em inglês; os dados
de hardware em `docs/` estão em português.

## License

MIT, see [LICENSE](LICENSE). To cite this project, use the "Cite this
repository" button on GitHub (`CITATION.cff`).
