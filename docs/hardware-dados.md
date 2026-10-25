# Dados de hardware para a simulação

Valores de hardware usados na simulação. "Não verificado" = valor de
loja ou derivado, não de folha de dados oficial.

## Servo Feetech STS3215

| Item | 7,4 V | 12 V |
|---|---|---|
| Torque de travamento | 19,5 kg·cm (1,91 N·m) | 30 kg·cm (2,94 N·m) |
| Velocidade sem carga | 0,192 s/60° (5,45 rad/s) | 0,222 s/60° (4,71 rad/s) |
| Corrente de travamento | 2,5 A | 2,7 A |
| Corrente em repouso | 150 mA | 200 mA |
| Faixa de tensão | 4 a 7,4 V | 6 a 12,6 V |

Fonte: [Waveshare ST3215](https://www.waveshare.com/st3215-servo.htm?sku=33014).

- Massa: 70 g (Waveshare) ou 55 g (lojas, não verificado). Usar 60 g até
  pesar um servo real.
- Dimensões: 45,2 × 24,7 × 35 mm (não verificado).
- Encoder magnético de 12 bits (4096 passos por volta), sem batente.
- Barramento serial TTL half-duplex, IDs 0 a 253, padrão 1 Mbps.
- Velocidade em modo roda: sem dado; usar a velocidade sem carga.

### Parâmetros MuJoCo usados por outros projetos

| Projeto | damping | frictionloss | armature | kp | kv | forcerange |
|---|---|---|---|---|---|---|
| [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground) (`open_duck_mini_v2.xml`, classe `sts3215`) | 0,56 | 0,068 | 0,027 | 13,37 | 0 | ±3,23 |
| [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (`Simulation/SO101/so101_new_calib.xml`) | 0,60 | 0,052 | 0,028 | 998,22 | 2,731 | ±3,35 |

Ponto de partida sugerido: valores do Open Duck (robô com pernas, mesmo uso),
forcerange reduzido para ±1,9 N·m na versão 7,4 V se o torque real for o da
tabela.

## Placas ESP32 para servos seriais

| Placa | Chip | Entrada | Tamanho | Preço |
|---|---|---|---|---|
| [Waveshare Servo Driver with ESP32](https://docs.waveshare.com/Servo_Driver_with_ESP32) | ESP32 (variante não informada) | 6 a 12 V | 65 × 30 mm | ~£15 |
| [Waveshare General Driver for Robots](https://www.waveshare.com/general-driver-for-robots.htm) | ESP32-WROOM-32UE | 7 a 13 V | 65 × 65 mm | ~US$ 28 |

- Driver ESP-IDF: componente [ftservo](https://components.espressif.com/components/dong-12345/ftservo).
- micro-ROS: ESP32-S3 testado com ESP-IDF 5.2 a 6.0
  ([micro_ros_espidf_component](https://components.espressif.com/components/micro-ros/micro_ros_espidf_component)).
- Circuito half-duplex para ESP32-S3 genérico: sem esquema documentado
  encontrado. Pendente.

## Outras massas

| Peça | Massa |
|---|---|
| Placa ESP32-S3 | ~20 g (não verificado) |
| LiPo 2S 1000 a 1500 mAh | pendente |
| Regulador step-down | pendente |

## Mola de TPU

Flexão passiva: lâmina de TPU com 12 mm de comprimento livre, presa entre dois
blocos de fixação. A rigidez à rotação usada na simulação é a de uma viga em
balanço:

k = E_TPU · (W · B³ / 12) / L

com W a largura da lâmina, B a espessura, L = 12 mm e
E_TPU = 26 MPa (não verificado). Amortecimento: 0,02 N·m·s/rad.

| Espessura B (mm) | Rigidez a W = 20 mm (N·m/rad) |
|---|---|
| 2 | 0,029 |
| 4 | 0,23 |
| 6 | 0,78 |
| 8 | 1,85 |

A rigidez cresce com o cubo da espessura: dobrar B multiplica k por oito.

## Open Duck Mini (referência de construção)

- Impressão em PLA com 15% de preenchimento; pés em TPU com 40%.
- 14 servos STS3215 de 7,4 V (não verificado); ~42 cm de altura.
- Coxa ~74 mm e canela ~79 mm (derivado do XML).
- Massa somada no XML: ~2,1 kg (parcial, não verificado).
