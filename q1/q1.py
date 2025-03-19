import cv2
import numpy as np

def detectar_formas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    vermelha = np.array([0, 120, 70])
    vermelha_max = np.array([10, 255, 255])
    azul = np.array([100, 150, 0])
    azul_max = np.array([140, 255, 255])
    
    verm_cont_m = cv2.inRange(hsv, vermelha, vermelha_max) 
    azul_cont_m = cv2.inRange(hsv, azul, azul_max)
    
    verm_cont_c, _ = cv2.findContours(verm_cont_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    azul_cont_c, _ = cv2.findContours(azul_cont_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return verm_cont_c, azul_cont_c

def maior_contorno(c):
    if c:
        return max(c, key=cv2.contourArea)
    return None

def obter_retangulo(c):
    if c is not None:
        return cv2.boundingRect(c)
    return None

def verificar_colisao(r1, r2):
    if r1 and r2:
        x1, y1, l1, a1 = r1
        x2, y2, l2, a2 = r2
        return not (x1 + l1 < x2 or x2 + l2 < x1 or y1 + a1 < y2 or y2 + a2 < y1)
    return False

def ultrapassou_barreira(r_maior, r_menor):
    if r_maior and r_menor:
        x1, y1, l1, a1 = r_maior
        x2, y2, l2, a2 = r_menor
        return x1 > x2 + l2 or x1 + l1 < x2
    return False

video = cv2.VideoCapture("q1A.mp4")
colisao_detectada = False

while True:
    sucesso, frame = video.read()
    if not sucesso:
        break
    
    verm_cont_c, azul_cont_c = detectar_formas(frame)
    
    verm_cont_m = maior_contorno(verm_cont_c)
    azul_cont_m = maior_contorno(azul_cont_c)
    
    verm_ret = obter_retangulo(verm_cont_m)  
    azul_ret = obter_retangulo(azul_cont_m)  
    
    if verm_ret: 
        x, y, l, a = verm_ret
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2) 
    
    if azul_ret:
        pass 
    
    if verificar_colisao(verm_ret, azul_ret):
        colisao_detectada = True
        cv2.putText(frame, "DETECTADA COLISAO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if colisao_detectada and ultrapassou_barreira(verm_ret, azul_ret):
        cv2.putText(frame, "PASSOU DA BARREIRA ", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Cor azul
    
    cv2.imshow("Feed", frame)
    
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:
        break
