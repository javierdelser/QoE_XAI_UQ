import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
from kan import KAN



def smape_loss(y_pred, y_true):
    """
    Computes the Symmetric Mean Absolute Percentage Error (SMAPE) loss between the predicted and true values.

    Args:
        y_pred (torch.Tensor): Predicted values of shape (batch_size, ...).
        y_true (torch.Tensor): True values of shape (batch_size, ...).

    Returns:
        torch.Tensor: SMAPE loss.
    """
    numerator = torch.abs(y_pred - y_true)
    denominator = (torch.abs(y_pred) + torch.abs(y_true)) / 2.0
    element_wise_smape = numerator / denominator
    return torch.mean(element_wise_smape).item()


class CustomDataset(Dataset): # se usa para organizar los datos en un formato que PyTorch pueda manejar fácilmente al entrenar redes neuronales
                              # Hereda de Dataset, que es la clase base para todos los datasets en PyTorch
    def __init__(self, feature_array, label_array):
        self.feature_array = feature_array # almacena las caracteristicas
        self.label_array = label_array # almacena las etiquetas

    def __len__(self):
        return len(self.feature_array) # Devuelve el número total de muestras (filas) en el dataset.
                                       # Esto es útil para que PyTorch sepa cuántas iteraciones tiene que hacer sobre los datos.

    def __getitem__(self, idx):
        return self.feature_array[idx, :], self.label_array[idx, :]  
        # Permite acceder a una muestra específica usando un índice (idx) - Devuelve una tupla (X[idx], y[idx])


class FullNet(nn.Module): # nn es el módulo que contiene las herramientas para construir redes neuronales
                        # nn.Linear() → Capa densa (fully connected).
                        # nn.ReLU() → Función de activación ReLU.
                        # nn.Embedding() → Capa de embeddings para datos categóricos.
                        # nn.L1Loss() → Función de pérdida L1 (MAE).
    def __init__(self, dic_num, dic_idx, dimension, device, base_dimension=128): # inicializacion
    # dic_num: Diccionario con el número de categorías únicas en cada variable categórica.
    # dic_idx: Diccionario con los índices de las columnas categóricas en los datos de entrada.
    # dimension: Número total de características en el dataset.
    # device: Define si se usa CPU o GPU.
    # base_dimension=128: Tamaño base para las capas lineales y embeddings.

    # FullNet hereda de nn.Module, lo que la convierte en una red neuronal personalizada en PyTorch.
    # super(FullNet, self) llama al constructor (__init__) de nn.Module.
    # Esto inicializa correctamente la clase base (nn.Module) y permite que FullNet herede sus funcionalidades.
        super(FullNet, self).__init__()
        '''
        How to set embedding size?
        https://forums.fast.ai/t/embedding-layer-size-rule/50691
        '''
        # EMBEDDING (declarar)
        # Transforman variables categóricas en vectores numéricos densos que la red puede usar para aprender patrones
        self.emb1 = nn.Embedding(dic_num['domain_name'], base_dimension)
        self.emb2 = nn.Embedding(dic_num['prov'], base_dimension)
        self.emb3 = nn.Embedding(dic_num['isp'], base_dimension)
        self.emb4 = nn.Embedding(dic_num['node_name'], base_dimension)
        self.emb5 = nn.Embedding(dic_num['id'], base_dimension)

        # FULLY CONNECTED LAYER
        # Se aplica una transformación lineal a las features numéricas
        self.first_linear = nn.Linear(dimension-5, base_dimension) # Se restan 5 porque hay 5 variables categóricas
        self.ratio = 0.15 # dropout - desactiva aleatoriamente un porcentaje de neuronas en cada forward pass (para evitar sobreajuste)

        # RED NEURONAL PRINCIPAL
        self.linear_relu_stack = nn.Sequential( # Automático: PyTorch maneja los gradientes, la optimización y la memoria.
            nn.Linear(base_dimension*10, base_dimension*5), # Cada capa oculta tiene menos neuronas para hacer aprendizaje progresivo
            nn.ReLU(), # introducir no linealidad - aprender relaciones complejas
            nn.Dropout(self.ratio),

            nn.Linear(base_dimension*5, base_dimension*2),
            nn.ReLU(),
            nn.Dropout(self.ratio),

            nn.Linear(base_dimension*2, base_dimension),
            nn.ReLU(),
            nn.Dropout(self.ratio),

            nn.Linear(base_dimension, 1), # La última capa tiene una única neurona, que da la predicción final
            # nn.ReLU()
        )
        self.dic_idx = [dic_idx[c] for c in ['domain_name', 'prov', 'isp', 'node_name', 'id']] # Obtiene los índices de las características categóricas (domain_name, prov, isp, node_name, id) en el dataset. Se usará en el forward() para manejar estas características con embeddings.
        self.device = device # Guarda si el modelo se ejecutará en CPU o GPU

    def forward(self, x): # forward() define cómo fluye la información a través del modelo cuando se hace una predicción
        len1 = x.shape[1] # x es el tensor de entrada, donde cada fila es un conjunto de características de una muestra

        # selecciona elementos específicos de un tensor a lo largo de un eje (input,dim(0 o 1 (columnas)),index)
        input_value = torch.index_select(x, 1, torch.tensor([i for i in range(len1) if i not in self.dic_idx]).to(self.device)) # selecciona todas las columnas que NO están en dic_idx
        input_emb = torch.index_select(x, 1, torch.tensor(self.dic_idx).to(self.device))

        emb1 = self.emb1(input_emb[:, 0].int()) # input_emb[:, 0] contiene los índices de la variable domain_name, que se pasan a self.emb1() para obtener el embedding correspondiente.
        emb2 = self.emb2(input_emb[:, 1].int())
        emb3 = self.emb3(input_emb[:, 2].int())
        emb4 = self.emb4(input_emb[:, 3].int())
        emb5 = self.emb5(input_emb[:, 4].int())

        iv = self.first_linear(input_value.float())

        # ELEMENT WISE PRODUCT (Creación del Vector de Entrada para la Red)
        input_revise = torch.cat([emb1*iv, emb2*iv, emb3*iv, emb4*iv, emb5*iv,
                                  emb1, emb2, emb3, emb4, emb5], 1)

        y = self.linear_relu_stack(input_revise)
        return y # prediccion final


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'checkpoint.pth')
        self.val_loss_min = val_loss


class Exp(object):

    def __init__(self, model, loss_fn, optimizer, scheduler, path, device):
        self.model = model
        self.path = path
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        if not os.path.exists(path):
            os.makedirs(path)

    def train_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """ Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

        Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A tuple of training loss and training accuracy metrics.
        In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
        """
        # Put model in train mode
        self.model.train()

        # Setup train loss and train accuracy values
        train_loss, train_smape = 0, 0

        # Loop through data loader data batches
        for batch, (X, y) in tqdm(enumerate(dataloader)): # tqdm - barra progreso
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)

            # 1. Forward pass (se llama a forward de FullNet)
            y_pred = self.model(X) # 𝑦=𝑊⋅𝑋+𝑏

            # 2. Calculate  and accumulate loss 
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimizer zero grad - borra los gradientes
            self.optimizer.zero_grad()

            # 4. Loss backward - Calcula el gradiente de la pérdida con respecto a los pesos y sesgos.
            loss.backward()

            # 5. Optimizer step - actualiza los pesos y sesgos del modelo en función de los gradientes calculados.
            self.optimizer.step()
            self.scheduler.step()

            # Calculate and accumulate accuracy metric across all batches
            # y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            # train_acc += (y_pred_class == y).sum().item()/len(y_pred)
            train_smape += smape_loss(y_pred, y)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader)
        train_smape = train_smape / len(dataloader)
        return train_loss, train_smape

    def test_step(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Tests a PyTorch model for a single epoch.

        Turns a target PyTorch model to "eval" mode and then performs
        a forward pass on a testing dataset.

        Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
        """
        # Put model in eval mode
        self.model.eval()

        # Setup test loss and test accuracy values
        test_loss, test_smape = 0, 0

        # Turn on inference context manager
        with torch.no_grad(): # Similar a train_step, pero sin actualizar pesos.
            # Loop through DataLoader batches
            for batch, (X, y) in tqdm(enumerate(dataloader)):
                # Send data to target device
                X, y = X.to(self.device), y.to(self.device)

                # 1. Forward pass
                y_pred = self.model(X)

                # 2. Calculate and accumulate loss
                loss = self.loss_fn(y_pred, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                # test_pred_labels = test_pred_logits.argmax(dim=1)
                # test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
                test_smape += smape_loss(y_pred, y)

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(dataloader)
        test_smape = test_smape / len(dataloader)
        return test_loss, test_smape

    def train(self, train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              epochs: int) -> Dict[str, List]:
        """Trains and tests a PyTorch model.

        Passes a target PyTorch models through train_step() and test_step()
        functions for a number of epochs, training and testing the model
        in the same epoch loop.

        Calculates, prints and stores evaluation metrics throughout.

        Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for
        each epoch.
        In the form: {train_loss: [...],
                      train_acc: [...],
                      test_loss: [...],
                      test_acc: [...]}
        For example if training for epochs=2:
                     {train_loss: [2.0616, 1.0537],
                      train_acc: [0.3945, 0.3945],
                      test_loss: [1.2641, 1.5706],
                      test_acc: [0.3400, 0.2973]}
        """
        # Create empty results dictionary
        results = {"train_loss": [], "train_smape": [], "test_loss": [], "test_smape": []}
        early_stopping = EarlyStopping(patience=50, verbose=True) # detiene el entrenamiento si no mejora
        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            start1 = datetime.now()
            train_loss, train_smape = self.train_step(dataloader=train_dataloader)
            start2 = datetime.now()
            test_loss, test_smape = self.test_step(dataloader=test_dataloader)
            start3 = datetime.now()

            print(
              f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_smape: {train_smape:.4f} | "
              f"train_time: {start2-start1} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_smape: {test_smape:.4f} | "
              f"test_time: {start3 - start2} | "
            )

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_smape"].append(train_smape)
            results["test_loss"].append(test_loss)
            results["test_smape"].append(test_smape)
            early_stopping(test_loss, self.model, self.path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        best_model_path = self.path + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return results

    def predict(self, test_dataloader: torch.utils.data.DataLoader):
        # Usa el modelo entrenado para hacer predicciones en datos nuevos.
        # No actualiza pesos.
        ret = []

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for batch, (X, _) in tqdm(enumerate(test_dataloader)):
                X = X.to(self.device)

                # 1. Forward pass
                test_pred = self.model(X)

                ret.append(test_pred)
        pred_tensor = torch.cat(ret, dim=0)
        return pred_tensor.detach().cpu().numpy()

# Modelo RVFL
class RVFL(BaseEstimator, RegressorMixin):
    def __init__(self, n_nodes, lam, w_random_range, b_random_range, activation, n_layer, same_feature=False,
                 random_state=None):
        
        # self te permite hacer referencia a los atributos y métodos de la clase dentro de sus propios métodos.
        self.n_nodes = n_nodes # nodos ocultos
        self.lam = lam # lambda (regularización) - controla cuánto se penalizan los pesos grandes
        self.w_random_range = w_random_range 
        self.b_random_range = b_random_range
        self.random_weights = [] # (n_feature, n_nodes) - Matriz de pesos aleatorios de la capa oculta.
        self.random_bias = [] # (1, n_nodes) - Vector de bias aleatorios para cada nodo oculto.
        self.beta = [] # (n_feature + n_nodes + 1, 1) - Matriz de pesos finales aprendidos por regresión lineal (formula, en vez de backpropagation)
        self.activation = activation
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.n_layer = n_layer
        self.data_std = [None]* self.n_layer
        self.data_mean = [None]* self.n_layer
        self.same_feature = same_feature
        self.random_state = random_state
        
    
    def fit(self, data, label): # X_train e y_train
        h = data.copy()
        n_sample = len(data)
        n_feature = len(data[0])
        data = self.standardize(data, 0)  
        y = label

        for i in range(self.n_layer):
            # normaliza la entrada a cada capa
            h = self.standardize(h, i)

            # genera pesos(W) y sesgo(b) aleatorios para cada capa
            self.random_weights.append(self.get_random_vector(len(h[0]), self.n_nodes, self.w_random_range))
            self.random_bias.append(self.get_random_vector(1, self.n_nodes, self.b_random_range))

            # Salida de la capa oculta - producto escalar y activación (RELU)
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]), self.random_bias[i]))
            
            # La salida de la capa oculta (h) se concatena con la entrada original (data) - Red Functional Link.
            # En lugar de pasar solo por una red profunda, la red usa tanto h como la data para predecir.
            d = np.concatenate([h, data], axis=1)
            h = d # Esto permite que la siguiente capa tenga acceso tanto a las salidas de la capa anterior como a las entradas originales, lo que refuerza la capacidad de aprendizaje del modelo.
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1) # Añade una columna de unos al final de la matriz - actúa como un bias global
            
            # Calcular beta (vector de pesos de salida)
            if n_sample > (self.n_nodes + n_feature):
                self.beta.append(np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y))
            else:
                self.beta.append(d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y))


    def predict(self, data):
        n_sample = len(data)
        h = data.copy()
        data = self.standardize(data, 0)  
        outputs = []

        for i in range(self.n_layer):
            h = self.standardize(h, i)  
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]), self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))
        return np.mean(outputs, axis=0)
    
    def eval_mae(self, data, label):
        n_sample = len(data)
        h = data.copy()
        data = self.standardize(data, 0)  
        outputs = []
        
        for i in range(self.n_layer):
            h = self.standardize(h, i)  
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]), self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))
            
        pred = np.mean(outputs, axis=0)
        mae = np.mean(np.abs(pred - label))
        return mae
    
    def eval_smape(self, data, label):
        n_sample = len(data)
        h = data.copy()
        data = self.standardize(data, 0)  
        outputs = []
        
        for i in range(self.n_layer):
            h = self.standardize(h, i)  
            h = self.activation_function(np.dot(h, self.random_weights[i]) + np.dot(np.ones([n_sample, 1]), self.random_bias[i]))
            d = np.concatenate([h, data], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))
            
        pred = np.mean(outputs, axis=0)

        epsilon = 1e-8
        smape = 100 * np.mean(2 * np.abs(pred - label) / (np.abs(pred) + np.abs(label) + epsilon))
        return smape
    
    def eval_mape(self, data, label):
        n_sample = len(data)
        h = data.copy()
        data = self.standardize(data, 0)  
        outputs = []
        
        for i in range(self.n_layer):
            h = self.standardize(h, i)  
            h = self.activation_function(
                np.dot(h, self.random_weights[i]) + 
                np.dot(np.ones([n_sample, 1]), self.random_bias[i])
            )
            d = np.concatenate([h, data], axis=1)
            h = d
            d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
            outputs.append(np.dot(d, self.beta[i]))
            
        pred = np.mean(outputs, axis=0)

        epsilon = 1e-8
        mape = 100 * np.mean(np.abs((label - pred) / (label + epsilon)))
        return mape

    
    @staticmethod
    def get_random_vector(m, n, scale_range):
        x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
        return x

    def standardize(self, x, index):
        if self.same_feature is True:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x)
            return (x - self.data_mean[index]) / self.data_std[index]
        else:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x, axis=0)
            return (x - self.data_mean[index]) / self.data_std[index]


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))

    @staticmethod
    def sine(x):
        return np.sin(x)

    @staticmethod
    def hardlim(x):
        return (np.sign(x) + 1) / 2

    @staticmethod
    def tribas(x):
        return np.maximum(1 - np.abs(x), 0)

    @staticmethod
    def radbas(x):
        return np.exp(-(x**2))

    @staticmethod
    def sign(x):
        return np.sign(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x):
        x[x >= 0] = x[x >= 0]
        x[x < 0] = x[x < 0] / 10.0
        return x

# ---WRAPPER---
# Clase que envuelve el modelo para que sea compatible con scikit-learn 
# Implementa los métodos fit, predict, get_params y set_params necesarios para RandomizedSearchCV o cross_val_score
class KANWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, width=[2, 5, 1], k=3, seed=42, lr=0.01, epochs=100, batch_size=64):
        self.width = width # arquitectura de la red [n_features_in, n_hidden_n, n_out]
        self.k = k # grado de los splines (curvas de activación)
        self.seed = seed
        self.lr = lr # tasa de aprendizaje
        self.epochs = epochs # epocas (ciclos completos sobre todos los datos)
        self.batch_size = batch_size # tamaño de cada batch de train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KAN(width=self.width, k=self.k, seed=self.seed).to(self.device)

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y * 100, dtype=torch.float32).reshape(-1, 1).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.L1Loss()

        self.model.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(X.size(0))
            for i in range(0, X.size(0), self.batch_size):
                idx = perm[i:i + self.batch_size]
                x_batch, y_batch = X[idx], y[idx]

                y_pred = self.model(x_batch)
                loss = criterion(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).cpu().numpy().flatten()
        return y_pred / 100
