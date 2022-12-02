import cv2
import numpy as np

#webcam = cv2.VideoCapture("http://192.168.15.16:4747/video")
webcam = cv2.VideoCapture(2, cv2.CAP_DSHOW)

camWidth = 1280
camHeight = 720
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, camWidth)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, camHeight)

model = "../model_data/frozen_inference_graph.pb"
configPath = "../model_data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
net = cv2.dnn_DetectionModel(model, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


with open('../model_data/coco.names', 'r') as f:
    classesList = f.read().splitlines()
    f.close()

classesList.insert(0, "__Background__")
classesList.remove("person")


if (webcam.isOpened()):
    frameCapturado, frame = webcam.read()

    while (frameCapturado):
        frameCapturado, frame = webcam.read()
        
        classLabelIDs, confidences, bboxes = net.detect(frame, confThreshold=0.4)
        bboxes = list(bboxes)
        confidences = list(np.array(confidences).reshape(1,-1)[0])
        confidences = list(map(float, confidences))
        
        bboxIdx = cv2.dnn.NMSBoxes(bboxes, confidences, score_threshold = 0.5, nms_threshold = 0.2)

        LastBoundBoxProdutcs = []
        if (len(bboxIdx)!=0):
            for i in range(0, len(bboxIdx)):
                bbox = bboxes[np.squeeze(bboxIdx[i])]
                classConfidence = confidences[np.squeeze(bboxIdx[i])]
                classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                classLabel = classesList[classLabelID]

                x,y,w,h = bbox

                LastBoundBoxProdutcs.append([x,y,w,h, None])

                cv2.rectangle(frame, (x,y), (x+w, y+h), color=(0,255,0), thickness=1)

                cv2.putText(frame, classLabel, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 2)
        
        cv2.imshow("videoSoruce", frame)



        if cv2.waitKey(10) ==ord("q"):
            webcam.release()
            cv2.destroyAllWindows()
            break