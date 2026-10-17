import numpy as np
import layer as l
class MLP:
    def __init__(self, Ws=None, bs=None):
        if Ws is None and bs is None:
            self.layers = [l.convo(1, 8), l.relu(), l.maxPool(), l.convo(8, 16), l.relu(), l.maxPool(), l.flatten(), l.Linear(784, 128), l.relu(), l.Linear(128, 10)]
        else:
            self.layers = [l.convo(1, 8, Ws[0], bs[0]), l.relu(), l.maxPool(), l.convo(8, 16, Ws[1], bs[1]), l.relu(), l.maxPool(),  l.flatten(), l.Linear(784, 10, Ws[2], bs[2]), l.relu(), l.Linear(128, 10, Ws[3], bs[3])]
        self.X_out = None
    def forward(self, X_in):
        for layer in self.layers:
            X_in = layer.forward(X_in)
        self.X_out = X_in
    def backward(self, G_out):
        for layer in reversed(self.layers):
            G_out = layer.backward(G_out)
    def update(self, alpha, t):
        for layer in self.layers:
            layer.update(alpha, t)
    def softmax(self, x):
            e_x = np.exp(x)
            return e_x/e_x.sum(axis=0)
    def predict(self, X_in):
        self.forward(X_in)
        p = self.softmax(self.X_out)
        return np.argmax(p)
    def loss(self, correct):
        p = self.softmax(self.X_out)
        L = np.empty(correct.size)
        one_hot = np.zeros(p.shape)
        for j in range(0, correct.size):
            L[j] = -np.log(p[correct[j], j])
            one_hot[correct[j], j] = 1    
        J = L.mean()
        G_out = p - one_hot
        return J, G_out
    def save_weight(self):
        Ws = []
        bs = []
        for layer in self.layers:
            W, b = layer.save_Weight()
            if W is None and b is None:
                pass
            else:
                Ws.append(W)
                bs.append(b)
        return Ws, bs
