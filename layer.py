import numpy as np
from scipy import signal
class Linear:
    def __init__(self, in_dim, out_dim, W=None, b=None):
        if W is None and b is None:
            self.W = np.random.randn(out_dim, in_dim)*np.sqrt(2.0/in_dim)
            self.b = np.zeros((out_dim, 1))
        else:
            self.W = W
            self.b = b
        self.X_in = None
        self.grad_W = None
        self.grad_b = None
        self.mW = 0
        self.mb = 0
        self.vW = 0
        self.vb = 0
    def forward(self, X_in):
        self.X_in = X_in

        return self.W@X_in + self.b 
    def backward(self, G_out):
        G_in = self.W.T@G_out
        self.grad_W = (G_out@self.X_in.T)/G_out.shape[1]
        self.grad_b = np.mean(G_out, axis=1, keepdims=True)
        return G_in
    def update(self, alpha, t):
        self.mW = 0.9*self.mW + 0.1*self.grad_W
        self.vW = 0.999*self.vW + 0.001*(self.grad_W*self.grad_W)
        self.mb = 0.9*self.mb + 0.1*self.grad_b
        self.vb = 0.999*self.vb + 0.001*(self.grad_b*self.grad_b)
        mW_b = self.mW/(1-0.9**t)
        vW_b = self.vW/(1-0.999**t)
        mb_b = self.mb/(1-0.9**t)
        vb_b = self.vb/(1-0.999**t)
        self.W-=(alpha*mW_b/(np.sqrt(vW_b) + 1e-8))
        self.b-=(alpha*mb_b/(np.sqrt(vb_b) + 1e-8))
    def save_Weight(self):
        return self.W, self.b
class relu:
    def __init__(self):
        self.RX_in = None
    def forward(self, X_in):
        self.RX_in = X_in > 0
        return np.maximum(0, X_in)
    def backward(self, G_out):
        return self.RX_in*G_out
    def update(self, alpha, t):
        pass
    def save_Weight(self):
            return None, None
class convo:
    def __init__(self, channels, filters, W=None, b=None):
        if W is None and b is None:
            self.W = np.random.randn(3,3, channels, filters)*np.sqrt(2.0/(channels*3*3))
            self.b = np.zeros((filters, 1))
        else:
            self.W = W
            self.b = b
        self.X_in = None
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.mW = 0
        self.vW = 0
        self.mb = 0
        self.vb = 0
    def convolution(self, X_in, W, b):
        X_out = np.zeros((X_in.shape[0], X_in.shape[1], W.shape[3], X_in.shape[3]))
        # make sure number of channels match
        for i in range(0, X_in.shape[3]):
            for j in range(0, W.shape[3]):
                sum = np.zeros((X_in.shape[0], X_in.shape[1]))
                for k in range(0, W.shape[2]):
                    sum+=(signal.correlate2d(X_in[:,:,k,i], W[:,:,k, j], mode="same"))
                X_out[:, :, j, i] = sum + b[j]
        return X_out
        
    def forward(self, X_in):
        self.X_in = X_in
        self.X_in = np.pad(self.X_in, pad_width=((1, 1), (1, 1), (0, 0), (0, 0)), mode="constant", constant_values=0)
        # remember to dropout for future projects that are more advanced
        return self.convolution(X_in, self.W, self.b)
    def backward(self, G_out):
        G_in = self.convolution(G_out, np.swapaxes(np.flip(self.W, axis=(0, 1)), axis1=2, axis2=3), np.zeros(( G_out.shape[3] , 1)))
        for k in range(0, G_out.shape[3]):
            for i in range(0,self.X_in.shape[2]):
                for j in range(0, G_out.shape[2]):
                    self.grad_W[:,:, i, j]+=(signal.correlate(self.X_in[:,:, i, k], G_out[:,:, j, k], mode= "valid")/G_out.shape[3])
        for i in range(0, G_out.shape[2]):
            self.grad_b[i, 0] = np.mean(G_out[:,:, i, :])
        return G_in
    def update(self, alpha, t):
        self.mW = 0.9*self.mW + 0.1*self.grad_W
        self.vW = 0.999*self.vW + 0.001*(self.grad_W*self.grad_W)
        self.mb = 0.9*self.mb + 0.1*self.grad_b
        self.vb = 0.999*self.vb + 0.001*(self.grad_b*self.grad_b)
        mW_b = self.mW/(1-0.9**t)
        vW_b = self.vW/(1-0.999**t)
        mb_b = self.mb/(1-0.9**t)
        vb_b = self.vb/(1-0.999**t)
        self.W-=(alpha*mW_b/(np.sqrt(vW_b) + 1e-8))
        self.b-=(alpha*mb_b/(np.sqrt(vb_b) + 1e-8))
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
    def save_Weight(self):
            return self.W, self.b
class maxPool:
    def __init__(self):
        self.RX_in = None
    def forward(self, X_in):
        self.RX_in = np.zeros_like(X_in)
        X_out = np.zeros((X_in.shape[0]//2, X_in.shape[1]//2, X_in.shape[2], X_in.shape[3]))
        for i in range(0, X_out.shape[0]):
            for j in range(0, X_out.shape[1]):
                X_out[i, j, :, :] = np.max(X_in[2*i:2*i+2, 2*j:2*j+2, :, :], axis=(0,1), keepdims=False)
                #Stretched = np.repeat(X_out[i, j, :, :], 2, axis=0)
                #Stretched =  np.repeat(Stretched, 2, axis=1)
                self.RX_in[2*i:2*i+2, 2*j:2*j+2, :, :] = X_in[2*i:2*i+2, 2*j:2*j+2, :, :] >= X_out[i, j, :, :]
                #self.RX_out[i, j, :, :] = np.argmax(X_in[2*i:2*i+2, 2*j:2*j+2, :, :], axis=(2, 3), keepdims=True)
        return X_out
    def backward(self, G_out):
        Stretched = np.repeat(G_out, 2, axis=0)
        Stretched = np.repeat(Stretched, 2, axis=1)
        return self.RX_in*Stretched
    def update(self, alpha, t):
        pass
    def save_Weight(self):
            return None, None
class flatten:
    def __init__(self):
        self.shape = None
    def forward(self, X_in):
        self.shape = X_in.shape
        return X_in.reshape(-1, X_in.shape[3])
    def backward(self, G_out):
        return G_out.reshape(self.shape)
    def update(self, alpha, t):
        pass
    def save_Weight(self):
            return None, None


                
