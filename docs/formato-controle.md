# Formato do arquivo de controle

Arquivo JSON gerado pelo export (Fase 4 de `plano-hardware.md`) e lido pelo
firmware do ESP32 (Fase 5). Descreve um robô já evoluído: quais servos
existem e o sinal que cada um segue. O firmware não precisa saber nada sobre
evolução, morfologia ou MuJoCo.

## Exemplo

```json
{
  "format": 1,
  "robot": "4legs-0wheels-8servos",
  "fitness": 1.23,
  "control_hz": 50,
  "ramp_time": 0.5,
  "frequency": 1.4,
  "servos": [
    {"id": 1, "name": "p0L_hip_pitch", "mode": "position",
     "amp": 0.35, "offset": -0.10, "phase": 1.57,
     "sign": 1, "zero": 2048, "min": -1.3, "max": 1.3},
    {"id": 2, "name": "p0L_knee", "mode": "position",
     "amp": 0.20, "offset": 0.30, "phase": 3.14,
     "sign": -1, "zero": 2048, "min": -1.3, "max": 1.3},
    {"id": 9, "name": "p1L_wheel", "mode": "wheel",
     "speed": 4.0, "sign": 1}
  ]
}
```

## Campos

| Campo | Tipo | Significado |
|---|---|---|
| `format` | int | versão deste formato; o firmware recusa versões que não conhece |
| `robot` | str | tipo do robô, igual a `replay.py --list` |
| `fitness` | float | só informativo |
| `control_hz` | int | frequência do laço de controle; a simulação usa a mesma (50) |
| `ramp_time` | float, s | tempo da rampa de partida, igual a `controller.RAMP_TIME` |
| `frequency` | float, Hz | frequência global do CPG |
| `servos` | lista | um item por servo, na ordem de envio |

Por servo:

| Campo | Modo | Significado |
|---|---|---|
| `id` | todos | ID do servo no barramento (1 a 253) |
| `name` | todos | junta na simulação (`p<par><L/R>_<junta>`), para montar e depurar |
| `mode` | todos | `position` (junta) ou `wheel` (rotação contínua) |
| `amp`, `offset`, `phase` | position | sinal em radianos: `offset + amp * sin(2π · frequency · t + phase)` |
| `min`, `max` | position | limites da junta em radianos; o firmware corta o alvo neles |
| `speed` | wheel | velocidade em rad/s (positiva anda para a frente) |
| `sign` | todos | `1` ou `-1`: corrige o sentido de montagem do servo |
| `zero` | position | posição do servo (0 a 4095) que corresponde a 0 rad; vem da calibração |

## Regras do firmware

A cada `1 / control_hz` segundos, com `t` = tempo desde a partida:

```
ramp = min(1, t / ramp_time)
alvo = ramp * (offset + amp * sin(2π · frequency · t + phase))   # position
alvo = clamp(alvo, min, max)
passos = zero + sign * alvo * 4096 / (2π)
vel = ramp * speed * sign                                          # wheel
```

É a mesma conta de `controller.CPG`, com o mesmo `ramp_time` e o mesmo
`control_hz` da simulação, para o robô real repetir o que foi evoluído.

## Calibração

`zero` e `sign` dependem de como cada servo foi montado e não saem da
simulação. O export grava `zero = 2048` e `sign` conforme o lado da perna; o
modo de calibração do firmware mede o `zero` real e atualiza o arquivo.
