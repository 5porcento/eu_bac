import cv2
import numpy as np

# Leitura da imagem
imagem = cv2.imread('Fotos/cultura_bacteriana.jpg')

# Conversão para escala de cinza
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicação de desfoque Gaussiano
desfocado = cv2.GaussianBlur(cinza, (9, 9), 2, 2)

# Detecção de círculos utilizando a Transformada de Hough
circulos = cv2.HoughCircles(desfocado, cv2.HOUGH_GRADIENT, dp=1, minDist=20,param1=50, param2=30, minRadius=0, maxRadius=0)

# Conversão dos círculos para inteiros (se algum círculo foi detectado)
if circulos is not None:
    circulos = np.round(circulos[0, :]).astype("int")

    # Desenho dos círculos detectados
    for (x, y, r) in circulos:
        cv2.circle(imagem, (x, y), r, (0, 255, 0), 4)

# Visualização da imagem com círculos detectados
cv2.imshow("Círculos Detectados", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()