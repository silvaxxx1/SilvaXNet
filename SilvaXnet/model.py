import cupy as cp

class NeuralNetwork:  
    def __init__(self):
        self._layers = []
        self._params = []
 
    def add_layer(self, layer):      
        self._layers.append(layer)
        if layer.params: 
            for i, _ in enumerate(layer.params):                         
                self._params.append([layer.params[i], layer.grads[i]])            
    
    def forward(self, X): 
        for layer in self._layers:
            X = layer.forward(X) 
        return X   

    def __call__(self, X):
        return self.forward(X)
    
    def predict(self, X):
        p = self.forward(X)       
        if p.ndim == 1:  # Single sample
            return cp.argmax(p)  
        return cp.argmax(p, axis=1)  # Multiple samples  
   
    def backward(self, loss_grad, reg=0.):
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i] 
            loss_grad = layer.backward(loss_grad)
            layer.reg_grad(reg) 
        return loss_grad
    
    def reg_loss(self, reg):
        reg_loss = 0
        for i in range(len(self._layers)):
            reg_loss += self._layers[i].reg_loss(reg)
        return reg_loss
    
    def parameters(self): 
        return self._params
    
    def zero_grad(self):
        for i, _ in enumerate(self._params):           
            self._params[i][1][:] = 0  # Reset gradients
            
    def get_parameters(self):
        return self._params   
    
    def save_parameters(self, filename):
        params = {}

        for i in range(len(self._layers)):
            if self._layers[i].params:
                params[i] = [cp.asnumpy(param) for param in self._layers[i].params]  # Convert to NumPy before saving
        cp.save(filename, params)               
        
    def load_parameters(self, filename):
        params = cp.load(filename, allow_pickle=True)
        count = 0
        for i in range(len(self._layers)):
            if self._layers[i].params:
                layer_params = [cp.array(param) for param in params.item().get(i)]  # Convert back to CuPy
                self._layers[i].params = layer_params                
                for j in range(len(layer_params)):                   
                    self._params[count][0] = layer_params[j]
