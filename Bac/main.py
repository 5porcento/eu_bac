import cv2
import numpy as np

imagem = cv2.imread('Fotos/2.jpg')
# Converter para escala de cinza
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
# Aplicar desfoque para reduzir o ruído
desfocada = cv2.GaussianBlur(gray, (5, 5), 0)

_, mascara = cv2.threshold(desfocada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Detecção de fundo
kernel = np.ones((3,3), np.uint8)
fundo = cv2.dilate(mascara, kernel, iterations=3)

# Detecção de primeiro plano
dist_transform = cv2.distanceTransform(mascara, cv2.DIST_L2, 5)
_, primeiro_plano = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Encontrar marcadores desconhecidos
primeiro_plano = np.uint8(primeiro_plano)
desconhecido = cv2.subtract(fundo, primeiro_plano)

# Marcadores
_, marcadores = cv2.connectedComponents(primeiro_plano)

# Adicionar 1 a todos os marcadores para garantir que o fundo seja 1 e não 0
marcadores = marcadores+1

# Marcar a região desconhecida com zero
marcadores[desconhecido==255] = 0

# Aplicar o Watershed
marcadores = cv2.watershed(imagem, marcadores)
imagem[marcadores == -1] = [255,0,0]

# Neste ponto, `imagem` contém os contornos segmentados

# Para visualizar
cv2.imshow('Imagem Segmentada', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Para salvar
cv2.imwrite('caminho_para_salvar_imagem_resultante.jpg', imagem)