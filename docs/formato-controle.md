# Formato do arquivo de controle

Arquivo JSON gerado por `experiments/export.py` e lido pelo firmware do
ESP32. Descreve um robô já evoluído: quais servos
existem e o controlador que cada um segue. O firmware não precisa saber nada
sobre evolução, morfologia ou MuJoCo.

Este é o formato 3, do controlador com dois harmônicos por junta, retorno de
carga (Tegotae) e comando de direção. O formato 2 continua sendo lido, com os
ganhos de direção iguais a zero. O formato 1 (um seno por junta, em malha
aberta) não é mais lido nem gerado.

## Exemplo

```json
{
  "format": 3,
  "robot": "8servos-mid-trot",
  "fitness": 0.11,
  "control_hz": 50,
  "ramp_time": 0.5,
  "load_full_scale": 0.7,
  "frequency": 1.31,
  "servos": [
    {"id": 1, "name": "s8_pitch", "mode": "position",
     "amp1": 0.11, "phase1": 2.38, "offset": -0.33,
     "amp2": 0.11, "phase2": 1.97, "sigma": 0.62,
     "steer_gain": 0.40, "steer_shift": 0.15,
     "sign": -1, "zero": 2048, "min": -1.3, "max": 1.3},
    {"id": 2, "name": "s4_roll", "mode": "position",
     "amp1": 0.20, "phase1": 5.71, "offset": 0.10,
     "amp2": 0.00, "phase2": 0.00, "sigma": -1.20,
     "steer_gain": -0.25, "steer_shift": -0.10,
     "sign": 1, "zero": 2048, "min": -0.9, "max": 0.9},
    {"id": 3, "name": "s9_wheel", "mode": "wheel",
     "speed": 4.0, "steer_gain": 0.0, "steer_shift": 2.5, "sign": 1}
  ]
}
```

## Campos

| Campo | Tipo | Significado |
|---|---|---|
| `format` | int | versão deste formato (3); o firmware também lê o formato 2, sem direção |
| `robot` | str | tipo do robô, igual a `replay.py --list` |
| `fitness` | float | só informativo |
| `control_hz` | int | frequência do laço de controle; a simulação usa a mesma (50) |
| `ramp_time` | float, s | tempo da rampa de partida, igual a `controller.RAMP_TIME` |
| `load_full_scale` | float | opcional, padrão `0.7`: fração do torque máximo do STS3215 que vale `N = 1`; é a `SERVO_TORQUE` da simulação (70% do torque de travamento) |
| `frequency` | float, Hz | frequência natural global; `omega = 2π · frequency` |
| `servos` | lista | um item por servo, na ordem do layout (a ordem do modelo; `id` = posição + 1) |

Por servo:

| Campo | Modo | Significado |
|---|---|---|
| `id` | todos | ID do servo no barramento (1 a 253) |
| `name` | todos | `s<vaga>_<eixo>`: vaga do módulo na árvore do corpo e eixo (`pitch`, `yaw`, `roll` ou `wheel`), para montar e depurar |
| `mode` | todos | `position` (junta) ou `wheel` (rotação contínua) |
| `amp1`, `phase1` | position | amplitude base (rad) e fase (rad) do primeiro harmônico, antes da direção |
| `amp2`, `phase2` | position | amplitude base (rad) e fase (rad) do segundo harmônico, antes da direção |
| `offset` | position | posição central base, em rad, antes da direção |
| `sigma` | position | ganho do retorno de carga, em rad/s (com sinal); `0` desliga o retorno |
| `steer_gain` | position | ganho de amplitude: o fator `1 + c * steer_gain` é cortado em `[0, 2]` |
| `steer_shift` | position | deslocamento do offset central com o comando `c`, em rad |
| `min`, `max` | position | limites da junta em radianos; o firmware corta o alvo neles |
| `speed` | wheel | velocidade base em rad/s (positiva anda para a frente), antes da direção |
| `steer_shift` | wheel | mudança da velocidade com o comando `c`, em rad/s, cortada em `WHEEL_RANGE` |
| `sign` | todos | `1` ou `-1`: corrige o sentido de montagem do servo |
| `zero` | position | posição do servo (0 a 4095) que corresponde a 0 rad; vem da calibração |

## Comando de direção

O comando `c` vem de fora do arquivo, de um controle remoto ou de um sensor, e
fica em `[-1, 1]`: `0` segue reto, `-1` vira para um lado e `+1` para o outro.
O firmware expõe `set_command(c)`, que corta o valor em `[-1, 1]` e recalcula os
alvos quando o comando muda. O arquivo traz `amp1`, `amp2`, `offset` e `speed`
sem a direção aplicada, então o mesmo `control.json` serve para qualquer
comando. O `steer_gain` da roda não é usado.

## Regras do firmware

Cada servo de posição `i` tem uma fase `theta_i`, estado do firmware, que
começa em 0 na partida. A cada tick, de período `dt = 1 / control_hz`, com
`t` = tempo desde a partida, e `omega = 2π · frequency`:

```
c       = clamp(comando, -1, 1)                   # 0 reto, -1 e +1 nos extremos
ganho   = clamp(1 + c * steer_gain, 0, 2)         # position
N       = sign * (carga_lida / carga_máxima) / load_full_scale   # cortado em -1 e 1
repetir 5 vezes, com h = dt / 5:                   # mesmos 4 ms da simulação
    dtheta = omega - sigma * N * cos(theta)       # Tegotae
    theta  = theta + h * dtheta
theta   = theta mod 2π                            # uma vez por tick, mantém a precisão do float

amp1    = amp1_base * ganho                       # position, antes do corte de velocidade
amp2    = amp2_base * ganho
offset  = offset_base + c * steer_shift
pico    = (amp1 + 2 * amp2) * omega
k       = min(1, 5.45 / pico)                     # corte pela velocidade do servo
amp1    = amp1 * k
amp2    = amp2 * k

ramp    = min(1, t / ramp_time)
alvo    = ramp * (offset + amp1 * sin(theta + phase1)
                         + amp2 * sin(2 * theta + phase2))
alvo    = clamp(alvo, min, max)
passos  = zero + sign * alvo * 4096 / (2π)

speed   = clamp(speed_base + c * steer_shift, -5.45, 5.45)   # wheel
vel     = ramp * speed * sign                     # wheel
```

- `carga_lida / carga_máxima` é a carga do servo dividida pelo torque máximo.
  A simulação normaliza pela fração `load_full_scale` desse torque (0,7), então
  a leitura é dividida por ela: 70% do torque máximo vira `N = 1`. No STS3215 vem do
  registro de carga presente (0x3C e 0x3D, 0 a 1000 em décimos de por cento do
  torque máximo, com um bit de direção). Multiplicar por `sign` leva a leitura
  para o sentido do eixo da junta, que é o sentido usado na simulação. Se o
  sinal da leitura ficar trocado em relação ao eixo, inverta-o aqui, não em
  `sigma`.
- Com `sigma = 0` a fase avança a `omega` e o controlador volta a ser um CPG em
  malha aberta. Com `sigma` diferente de zero, uma perna que sustenta peso
  atrasa ou adianta a própria fase conforme a carga (Tegotae, Owaki et al.), e
  as pernas se coordenam sem acoplamento programado entre elas.
- A fase é integrada em 5 subpassos de 4 ms por tick, como na simulação
  (passo de física de 4 ms, alvo recalculado a cada 20 ms). A carga `N` é lida
  uma vez por tick e usada nos 5 subpassos; na simulação ela é atualizada a
  cada 4 ms, diferença pequena porque a carga varia devagar.
- O arquivo traz as amplitudes e offsets decodificados, antes da direção e do
  corte de velocidade. O firmware aplica `c * steer_gain` e `c * steer_shift`
  e só então refaz o corte `(amp1 + 2 * amp2) * omega <= 5,45 rad/s`, com as
  duas amplitudes reduzidas na mesma proporção quando passam disso.
- Nas rodas o `steer_shift` é uma mudança de velocidade em rad/s, cortada em
  `WHEEL_RANGE` (a velocidade máxima do servo, com sinal). Rodas ignoram fase e
  carga: só `speed`, com a direção e a rampa.
- O `steer_gain` da roda não é usado, mas o campo é gravado para o formato ser
  uniforme; arquivos do formato 2 são lidos com `steer_gain = steer_shift = 0`.
- É a mesma conta de `controller.py` (`phase_rate`, `servo_targets`), com o
  mesmo `ramp_time` e o mesmo `control_hz` da simulação, para o robô real
  repetir o que foi evoluído.

## Calibração

`zero` e `sign` dependem de como cada servo foi montado e não saem da
simulação. O export grava `zero = 2048` e `sign` conforme o eixo da junta
(`-1` quando o módulo é espelhado) e o lado da roda; o modo de calibração do
firmware mede o `zero` real, na pose reta do robô, e atualiza o arquivo.
