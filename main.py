#Instancias
import cv2 
import time
import numpy as np

#cores
COLORS = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)]

#carregando as classes
class_names = []
with open("coco.names","r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#reprodução do vídeo
cap = cv2.VideoCapture("teste.mp4")

#carregando pesos nas redes neurais
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

#configurando os parametros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (416,416), scale = 1/255)

#leitura dos frames de um vídeo
while True:

    #captura de frame
    _, frame = cap.read()

    #contagem do FPS
    start = time.time()

    #detecção 
    classes, scores, boxes = model.detect(frame,0.01,0.02)

    #fim da contagem do fps
    end = time.time()


    #percorrer detecções
   # Percorrer detecções
    for (classid, score, box) in zip(classes, scores, boxes):
        # Verificar se o índice é válido em class_names
        if classid < len(class_names):
            # Obter o nome da classe
            class_name = class_names[classid]
            
            # Verificar se é um animal
            if class_name in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
                # Gerar uma cor para cada classe
                color = COLORS[int(classid) % len(COLORS)]

                # Obter o nome da classe e o score
                label = f"{class_name}: {score}"

                # Desenhar a caixa de detecção
                cv2.rectangle(frame, box, color, 2)

                # Escrever o nome da classe acima da caixa
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #calculando o tempo que levou para fazer a detecção 
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"

    #escrevendo o fps
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    #mostrando a imagem
    cv2.imshow("detections", frame)

    #espera a resposta
    if cv2.waitKey(1) == 27:
        break

#liberação da camera e destroi as janelas
cap.release()
cv2.destroyAllWindows()#Instancias
import cv2 
import time
import numpy as np

#cores
COLORS = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)]

#carregando as classes
class_names = []
with open("coco.names","r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

#reprodução do vídeo
cap = cv2.VideoCapture("teste.mp4")

#carregando pesos nas redes neurais
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

#configurando os parametros da rede neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (416,416), scale = 1/255)

#leitura dos frames de um vídeo
while True:

    #captura de frame
    _, frame = cap.read()

    #contagem do FPS
    start = time.time()

    #detecção 
    classes, scores, boxes = model.detect(frame,0.1,0.2)

    #fim da contagem do fps
    end = time.time()


    #percorrer detecções
    for(classid, score, box) in zip(classes, scores, boxes):

        #gerando uma cor para cada classe
        color = COLORS[int(classid) % len(COLORS)]

        #pegando o nome da classe pelo id e seu score
        label = f"{class_names[classid[0]]} : {score}"

        #desenhando box de detecção 
        cv2.rectangle(frame, box, color, 2)

        #escrevendo o nome da class em cima do box
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #calculando o tempo que levou para fazer a detecção 
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"

    #escrevendo o fps
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    #mostrando a imagem
    cv2.imshow("detections", frame)

    #espera a resposta
    if cv2.waitKey(1) == 27:
        break

#liberação da camera e destroi as janelas
cap.release()
cv2.destroyAllWindows()