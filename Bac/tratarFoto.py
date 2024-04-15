import cv2
import numpy as np

def tratamento_imagem(imagem_path):
    imagem = cv2.imread(imagem_path)

    alfa = 1.5
    beta = 0
    imagem_melhorada = cv2.convertScaleAbs(imagem, alpha=alfa, beta=beta)
    # filtro Gaussiano para suavizar a imagem
    imagem_suavizada = cv2.GaussianBlur(imagem_melhorada, (5, 5), 0)

    # Detecção de bordas com Canny
    bordas = cv2.Canny(imagem_suavizada, 100, 200)

    # Encontrar contornos
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagem_contornos = imagem.copy()

    # Filtrar contornos que se aproximam de um círculo
    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        area = cv2.contourArea(contorno)
        if perimetro == 0:
            continue
        circularidade = 4 * np.pi * (area / (perimetro * perimetro))

        # Considerar como placa de Petri se a circularidade estiver próxima de 1
        if 0.7 < circularidade < 1.2:
            # Encontrar o círculo mínimo que cobre o contorno para desenhar
            (x, y), raio = cv2.minEnclosingCircle(contorno)
            centro = (int(x), int(y))
            raio = int(raio)
            cv2.circle(imagem_contornos, centro, raio, (0, 255, 0), 2)

    largura_desejada = 1280
    altura_desejada = 720
    imagem_final = cv2.resize(imagem_contornos, (largura_desejada, altura_desejada))

    return imagem_final


imagem_processada = tratamento_imagem('Fotos/Captura de tela 2024-04-13 001146.png')
cv2.imshow('Imagem Processada', imagem_processada)
cv2.waitKey(0)
cv2.destroyAllWindows()
