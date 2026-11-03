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
- Encoder magnético de 12 bits (4096 passos por volta), sem batente.
- Barramento serial TTL half-duplex, IDs 0 a 253, padrão 1 Mbps.
- Velocidade em modo roda: sem dado; usar a velocidade sem carga.

### Medidas do corpo e da furação

Vêm de duas fontes independentes que concordam no que importa para encaixar
e parafusar: o modelo STEP do projeto [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(`STEP/SO100/STS3215_03a.step`, 2025, Apache-2.0) e o STEP e o desenho 2D da
[Waveshare](https://www.waveshare.com/wiki/ST3215_Servo) (ST3215, mesmo corpo,
2022). `darwinbot/parts.py` usa essas medidas; ainda não foram conferidas com
paquímetro num servo real. Origem no centro do corpo, x ao longo do corpo, z ao
longo do eixo de saída. As duas fontes divergem só na altura (até 2 mm) e no
diâmetro do horn (0,8 mm); a tabela traz as duas quando divergem.

| Item | Valor |
|---|---|
| Corpo (x × y) | 45,4 × 24,8 mm (Waveshare: 45,22 × 24,72) |
| Altura do corpo, sem horns | 30,7 mm, faces em z = -15,5 e +15,2 (Waveshare: 32) |
| Altura com os dois horns | 39,6 mm com o eixo saliente (Waveshare: 37,25); 35 mm com um horn só |
| Eixo de saída | a +12,5 mm do centro do corpo, nos dois lados (eixo duplo) |
| Disco do horn | Ø20 mm (Waveshare: Ø19,2), 2,2 mm de espessura, um em cada face |
| Furação do horn | 4 furos Ø2,5 mm num quadrado de 9,9 mm (círculo de Ø14,0 mm), a 45° dos eixos do corpo, mais um furo central Ø2,5 mm para o parafuso |
| Furação de fixação | 4 furos M2 roscados por face (topo e base), em x = +5,0 e -19,5 (base) ou -15,7 (topo), y = ±10,2 (Waveshare: ±10,25); o corpo não tem orelhas |

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
- Circuito half-duplex para ESP32-S3 genérico: ver a seção abaixo.

### Circuito half-duplex

A UART do ESP32 tem um fio para enviar (TX) e outro para receber (RX); o
barramento dos servos tem um só. Opções:

| Opção | Como é | Prós | Contras |
|---|---|---|---|
| A. Buffer de três estados com pino de direção | Um 74LVC1G125 ou 74HC126 liga o TX ao barramento só enquanto o ESP32 transmite; um pino do ESP32 controla a direção | Circuito clássico dos servos Dynamixel, confiável a 1 Mbps | Errar o momento de soltar o barramento corrompe a resposta |
| B. A mesma, com a direção automática | O modo `UART_MODE_RS485_HALF_DUPLEX` do ESP-IDF controla o pino de direção pelo hardware | Tira o risco de tempo da opção A | Ainda precisa do chip |
| C. Resistor ou diodo, sem chip | TX ligado ao fio por ~1 kΩ ou diodo, RX direto no fio | Quase de graça | Frágil a 1 Mbps; depende do cabo e do número de servos |
| D. Placa pronta | Waveshare Servo Driver with ESP32 | Funciona de cara | ESP32 comum, não o S3 |

Recomendada: B. Falta verificar o nível de tensão do sinal do STS3215 (3,3 V ou
5 V); se for 5 V, usar o 74HCT126, que converte o nível. Cada servo precisa de
um ID próprio antes de ir para o barramento (todos saem de fábrica com o mesmo).

### Por que servos de barramento, e não PWM

Servos de hobby comuns (SG90, MG996R) recebem um fio de PWM cada um, em geral
de uma placa como a PCA9685, e não devolvem nada. O controlador precisa da
carga de cada servo a cada ciclo (o retorno de carga da CPG), e só um servo de
barramento como o STS3215 a informa. O ESP32 fala direto com os servos: todos
ligados em série num único fio de dados, cada um com seu ID, sem placa de
controle no meio.

### Tempo do barramento (estimativa, não medida)

A 1 Mbps cada byte leva cerca de 10 µs. Por ciclo de controle (20 ms a 50 Hz):

| Parte | Uma linha, 16 servos | Duas linhas, 8 + 8 |
|---|---|---|
| Enviar os alvos num comando só (`SYNC WRITE`) | ~1,2 ms | ~0,6 ms em cada, ao mesmo tempo |
| Ler posição e carga, servo por servo | ~4 a 8 ms | ~2 a 4 ms |
| Total | ~5 a 9 ms | ~3 a 5 ms |

A 50 Hz uma linha deve bastar. O ESP32-S3 tem 3 UARTs, então duas linhas são
possíveis, ao custo de um segundo circuito half-duplex; valem a pena para
controle a 100 Hz, para ler mais dados por servo ou se a resposta do servo for
mais lenta que o previsto. Medir primeiro com uma linha.

Falta verificar: o tempo de espera antes da resposta (*return delay*) do
STS3215 e se o firmware do servo aceita `SYNC READ` (ler vários servos num
comando), que encurtaria a leitura.

## Outras massas

| Peça | Massa |
|---|---|
| Berço impresso do servo (PLA) | ~23 g (18,3 cm³ maciço, com o pé que parafusa na placa do tronco), somado aos 60 g do servo na simulação; a ponta do link e o fim do cotovelo trazem o berço do servo seguinte |
| Placa ESP32-S3 | ~20 g (não verificado) |
| LiPo 2S 1000 a 1500 mAh | pendente |
| Regulador step-down | pendente |

## Berço, link em U e cunha

O servo fica num berço impresso que pertence à peça-mãe: uma luva em volta do corpo (folga de 0,4 mm por lado) com uma tampa em cada face dos horns, onde os quatro parafusos M2 entram nos furos roscados do servo. Na placa do tronco o berço tem um pé com o padrão M3 comum; na ponta de um link ele sai numa peça só com o link, e a placa de cotovelo termina no berço do módulo filho. O link do próprio servo começa num garfo em U que abraça os dois horns (um braço do garfo em cada face, com o padrão de furos do horn) e só vira placa depois de passar o berço, para girar toda a faixa da junta.

Quando o slot do tronco tem rolagem que não é múltiplo de 90 graus (até 25), o pé fica inclinado e uma cunha impressa (`wedge_{graus}`) entre a aba e o pé compensa. Encaixe em face do tronco e placa de coluna nunca levam cunha. Eixo de saída na vertical parafusa por trás do servo (`_back`); encaixe em face, junta de rolagem com o corpo atravessando a placa e rolagem acima de 25 graus saem sem pé (`_free`).

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
