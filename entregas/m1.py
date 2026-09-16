import cv2
import numpy as np
import os

diretorio_atual = os.path.dirname(os.path.abspath(__file__))

fps = 20
largura = 640
altura = 480
n_quadros = 60

video_nome = os.path.join(diretorio_atual, "video_m1.mp4")
frame_nome = os.path.join(diretorio_atual, "frame_meio.jpg")

# 1 - criacao do video

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

escritor = cv2.VideoWriter(
    video_nome,
    fourcc,
    fps,
    (largura, altura)
)

for quadro_atual in range(n_quadros):

    quadro = np.zeros(
        (altura, largura, 3),
        dtype=np.uint8
    )

    centro = (largura // 2, altura // 2)

    x_circulo = int(
        80 + (largura - 160) * quadro_atual / (n_quadros - 1)
    )

    cv2.circle(
        quadro,
        (x_circulo, centro[1]),
        40,                 
        (0, 255, 255),      
        -1                  
    )

    x_retangulo = int(
        (largura - 160) * (1 - quadro_atual / (n_quadros - 1))
    )

    cv2.rectangle(
        quadro,
        (x_retangulo, 80),
        (x_retangulo + 80, 140),
        (255, 0, 0),        
        3                   
    )

    cv2.drawMarker(
        quadro,
        centro,
        (0, 255, 0),
        cv2.MARKER_CROSS,
        30,
        2
    )

    cv2.putText(
        quadro,
        f"Quadro: {quadro_atual + 1}/{n_quadros}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    escritor.write(quadro)

escritor.release()

# 2 - abertura do video

cap = cv2.VideoCapture(video_nome)

print("Vídeo aberto:", cap.isOpened())

fps_video = cap.get(cv2.CAP_PROP_FPS)
total_quadros = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
largura_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps_video)
print("Total de quadros:", total_quadros)
print("Tamanho:", f"{largura_video}x{altura_video}")

quadros = []

while True:

    ret, quadro = cap.read()

    if not ret:
        break

    quadros.append(quadro)

cap.release()

# 3  salvar o frame do meio

total_lidos = len(quadros)

indice_meio = total_lidos // 2

cv2.imwrite(
    frame_nome,
    quadros[indice_meio]
)

print("Quadros lidos:", total_lidos)
print("Índice do quadro do meio:", indice_meio)
print("Quadro salvo:", frame_nome)
print("Vídeo criado:", video_nome)


# O video é representado coomo uma sequencia de imagens exibidas a uma taxa de quadros por segundo chamada FPS 
# cada quadro é uma grade de pixes onde cada pixel possui tres canais de cor (azul, verdde e vermelho) que definem sua cor.
