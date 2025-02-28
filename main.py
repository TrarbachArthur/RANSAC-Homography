# TRABALHO 2 DE VISÃO COMPUTACIONAL
# Nome: Arthur Trarbach Sampaio

# Importa as bibliotecas necessárias
# Acrescente qualquer outra que quiser
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv


########################################################################################################################
# Função para normalizar pontos
# Entrada: points (pontos da imagem a serem normalizados)
# Saída: norm_points (pontos normalizados)
#        T (matriz de normalização)
def normalize_points(points):

    # Calculate centroid
    centroid = np.mean(points[0:2, :], axis=0)
    # Calculate the average distance of the points having the centroid as origin
    avg_distance = np.mean(np.linalg.norm(points - centroid, axis=1))
    # Define the scale to have the average distance as sqrt(2)
    scale = np.sqrt(2) / avg_distance
    # Define the normalization matrix (similar transformation)
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    # Normalize points

    pts = np.hstack((points, np.ones((len(points), 1))))

    norm_pts = (T @ pts.T).T[:, :2]
    
    return norm_pts, T

# Função para montar a matriz A do sistema de equações do DLT
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: A (matriz com as duas ou três linhas resultantes da relação pts2 x H.pts1 = 0)
def compute_A(pts1, pts2):
    
    # Add homogeneous coordinates
    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1)) )).T
    pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1)) )).T

    # Compute matrix A
    npoints = pts1.shape[1]
    A = np.zeros((3*npoints,9))

    for k in range(npoints):
        A[3*k, 3:6] = -pts2[2,k]*pts1[:,k]
        A[3*k, 6:9] = pts2[1,k]*pts1[:,k]

        A[3*k+1, 0:3] = pts2[2,k]*pts1[:,k]
        A[3*k+1, 6:9] = -pts2[0,k]*pts1[:,k]

        A[3*k+2, 0:3] = -pts2[1,k]*pts1[:,k]
        A[3*k+2, 3:6] = pts2[0,k]*pts1[:,k]

    return A

# Função do DLT Normalizado
# Entrada: pts1, pts2 (pontos "pts1" da primeira imagem e pontos "pts2" da segunda imagem que atendem a pts2=H.pts1)
# Saída: H (matriz de homografia estimada)
def compute_normalized_dlt(pts1, pts2):

    # Normaliza pontos

    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)

    # Constrói o sistema de equações empilhando a matrix A de cada par de pontos correspondentes normalizados

    A = compute_A(norm_pts1, norm_pts2)

    # Calcula o SVD da matriz A_empilhada e estima a homografia H_normalizada 

    U,S,Vt = np.linalg.svd(A)

    # Reshape last column of V as the homography matrix

    h = Vt[-1]
    H_norm = np.reshape(h,(3,3))

    # Mantendo parecido com o OpenCV
    H_norm = H_norm/H_norm[2,2]

    # Denormaliza H_normalizada e obtém H

    H = np.linalg.inv(T2) @ H_norm @ T1

    # Mantendo parecido com o OpenCV
    H = H / H[2,2] 

    return H


# Função do RANSAC
# Entradas:
# pts1: pontos da primeira imagem
# pts2: pontos da segunda imagem 
# dis_threshold: limiar de distância a ser usado no RANSAC
# N: número máximo de iterações (pode ser definido dentro da função e deve ser atualizado 
#    dinamicamente de acordo com o número de inliers/outliers)
# Ninl: limiar de inliers desejado (pode ser ignorado ou não - fica como decisão de vocês)
# Saídas:
# H: homografia estimada
# pts1_in, pts2_in: conjunto de inliers dos pontos da primeira e segunda imagens


def RANSAC(pts1, pts2, dis_threshold, N, Ninl):
    
    # Define outros parâmetros como número de amostras do modelo, probabilidades da equação de N, etc 
    max_in_cnt = 0
    best_pts1_in = None
    best_pts2_in = None
    s = 4

    i = 0
    # Processo Iterativo
    # Enquanto não atende a critério de parada
    while i < N:
        # Sorteia aleatoriamente "s" amostras do conjunto de pares de pontos pts1 e pts2 
        pts_indexes = np.random.choice(len(pts1), s, replace=False)
        temp_pts1 = pts1[pts_indexes]
        temp_pts2 = pts2[pts_indexes]
        
        # Usa as amostras para estimar uma homografia usando o DTL Normalizado
        H_temp = compute_normalized_dlt(temp_pts1, temp_pts2)

        # Testa essa homografia com os demais pares de pontos usando o dis_threshold e contabiliza
        # o número de supostos inliers obtidos com o modelo estimado

        # Add coordenadas homogeneas
        pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1)) )).T
        # Gerando pontos na segunda imagem, com base nos pontos da primeira
        pts2_gen = (H_temp @ pts1_h).T
        pts2_gen = pts2_gen / pts2_gen[:, 2][:, np.newaxis]
        pts2_gen = pts2_gen[:, :2]

        # Calculando distancias entre os pontos dados e os pontos calculados na segunda imagem
        distances = np.linalg.norm(pts2 - pts2_gen, axis=1)

        # inliers é um vetor de 0's e 1's onde 1 significa que o ponto do indice é um inlier
        inliers = distances < dis_threshold
        
        # Se o número de inliers é o maior obtido até o momento, guarda esse conjunto além das "s" amostras utilizadas. 
        # Atualiza também o número N de iterações necessárias
        # Número de inliers é definido pela soma dos valores no vetor (1 é inliers, 0 não é)
        if np.sum(inliers) > max_in_cnt:
            max_in_cnt = len(inliers)
            best_pts1_in = pts1[inliers]
            best_pts2_in = pts2[inliers]

            e = 1 - (np.sum(inliers)/len(pts1))
            N = (np.log(1-.99)//np.log(1-((1-e)**s))) + 1

        i += 1

    # Terminado o processo iterativo
    # Estima a homografia final H usando todos os inliers selecionados.
    pts1_in = best_pts1_in
    pts2_in = best_pts2_in

    H = compute_normalized_dlt(pts1_in, pts2_in)

    return H, pts1_in, pts2_in

########################################################################################################################
# Exemplo de Teste da função de homografia usando o SIFT


MIN_MATCH_COUNT = 10
img1 = cv.imread('comicsStarWars01.jpg', 0)   # queryImage
img2 = cv.imread('comicsStarWars02.jpg', 0)        # trainImage

# Inicialização do SIFT
sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])
    
    #################################################
    M, pts1_in, pts2_in = RANSAC(src_pts, dst_pts, 10, 1000, 0)
    #################################################

    img4 = cv.warpPerspective(img1, M, (img2.shape[1], img2.shape[0])) 

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

fig, axs = plt.subplots(2, 2, figsize=(30, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(img3, 'gray')
fig.add_subplot(2, 2, 2)
plt.title('Primeira imagem')
plt.imshow(img1, 'gray')
fig.add_subplot(2, 2, 3)
plt.title('Segunda imagem')
plt.imshow(img2, 'gray')
fig.add_subplot(2, 2, 4)
plt.title('Primeira imagem após transformação')
plt.imshow(img4, 'gray')
plt.show()

########################################################################################################################
