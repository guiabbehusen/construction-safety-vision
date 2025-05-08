import torch
import torch.nn as nn
import numpy as np

# Definição do modelo LSTM (igual ao seu código principal)
class AccidentLSTM(nn.Module):
    def __init__(self):
        super(AccidentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=34, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Criar instância do modelo
model = AccidentLSTM()

# Criar dados fictícios (10 sequências de 10 frames com 34 features)
X_dummy = torch.randn(10, 10, 34)  # batch_size=10, sequence_len=10, features=34
y_dummy = torch.randint(0, 2, (10,))  # classes 0 ou 1

# Definir perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Treinamento rápido (5 épocas)
for epoch in range(5):
    outputs = model(X_dummy)
    loss = criterion(outputs, y_dummy)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Salvar o modelo treinado
torch.save(model.state_dict(), "lstm_accident_model.pt")
print("Modelo salvo como lstm_accident_model.pt")
