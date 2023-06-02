import cv2 
import numpy as np
from IPython.display import display, clear_output
from google.colab.patches import cv2_imshow
import time

#cores
COLORS = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)]

#carregando as classes
class_names = []
with open("/content/sample_data/coco.names","r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#Definição dos rótulos utilizados  
animais = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

#reprodução do vídeo
cap = cv2.VideoCapture("/content/sample_data/teste.mp4")

#erro ao tentar abrir o video
if not cap.isOpened():
    print("Erro ao abrir o arquivo de vídeo.")
    exit()

#carregando pesos nas redes neurais
net = cv2.dnn.readNet("/content/sample_data/yolov7-tiny.weights", "/content/sample_data/yolov7-tiny.cfg")

#configurando os parametros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (416,416), scale = 1/255)

#leitura dos frames de um vídeo
while True:
    #captura de frame
    ret, frame = cap.read()
    frame_redimensionado = cv2.resize(frame, (700, 480))

    # Verifica se o quadro foi lido corretamente
    if not ret:
        # Se não, o vídeo acabou ou ocorreu um erro
        break

    #contagem do FPS
    start = time.time()
    
    #detecção 
    classes, scores, boxes = model.detect(frame_redimensionado,0.01,0.02)

    #fim da contagem do fps
    end = time.time()

    #Percorrer detecções
    for (classid, score, box) in zip(classes, scores, boxes):
        # Verificar se o índice é válido em class_names
        if classid < len(class_names):
            # Obter o nome da classe
            class_name = class_names[classid]
            if class_name in animais and score*100>=25:
            # Gerar uma cor para cada classe
                color = COLORS[int(classid) % len(COLORS)]

                # Obter o nome da classe e o score
                label = f"ID:{classid} - {class_name}: {(score*100):.2f}%"

                # Desenhar a caixa de detecção
                cv2.rectangle(frame_redimensionado, box, (0, 0, 0), 5)
                cv2.rectangle(frame_redimensionado, box, color, 4)

                # Escrever o nome da classe acima da caixa
                cv2.putText(frame_redimensionado, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(frame_redimensionado, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #calculando o tempo que levou para fazer a detecção 
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"

    #escrevendo o fps
    cv2.putText(frame_redimensionado, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame_redimensionado, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 3)

    # --------------------------------- Em algum amgiente jupyter que suporte o cv2.imshow() como o vsCode pode se utilizar:
    # --------------------------------- v2.imshow("Video Reconhecimento - PRESSIONE 'q' PARA FECHAR", frame)
    # --------------------------------- No lugar das 3 linhas abaixo
    #mostrando a imagem
    cv2_imshow(frame_redimensionado)
    time.sleep(0.15)
    #Limpando buffer para não empilhar os frames
    clear_output(True)

cap.release()
cv2.destroyAllWindows()#Instancias
