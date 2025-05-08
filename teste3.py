import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import torch
import torch.nn as nn
import os

output_dir = r"C:\Users\User\Documents\GitHub\construction-safety-vision\teste_estruturas2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

yolo_model = YOLO("yolov8m.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

class AccidentLSTM(nn.Module):
    def __init__(self):
        super(AccidentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=34, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

lstm_model_path = "lstm_accident_model.pt"
if not os.path.exists(lstm_model_path):
    raise FileNotFoundError(f"Modelo LSTM não encontrado em {lstm_model_path}. Treine o modelo primeiro.")
lstm_model = AccidentLSTM()
lstm_model.load_state_dict(torch.load(lstm_model_path))
lstm_model.eval()

path = r"C:\Users\User\Documents\GitHub\construction-safety-vision\videos\video5.mp4"
cap = cv2.VideoCapture(path)

if not cap.isOpened():
    raise FileNotFoundError(f"Não foi possível abrir o vídeo em {path}. Verifique o caminho.")

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = os.path.join(output_dir, "output5.mp4")
out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

sequence_length = 10
sequences = {}

save_sequences = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.track(frame, persist=True, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)
            pose_features = np.zeros(32)
            if pose_results.pose_landmarks:
                for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    if i < 16:
                        pose_features[2*i] = landmark.x
                        pose_features[2*i+1] = landmark.y

            features = np.concatenate(([cx, cy], pose_features))
            if track_id not in sequences:
                sequences[track_id] = []
            sequences[track_id].append(features)
            if len(sequences[track_id]) > sequence_length:
                sequences[track_id].pop(0)

            if len(sequences[track_id]) == sequence_length:
                sequence = np.array(sequences[track_id])[np.newaxis, ...]
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
                with torch.no_grad():
                    output = lstm_model(sequence_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                if prediction == 1:
                    cv2.putText(frame, "ACIDENTE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Detecção", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
