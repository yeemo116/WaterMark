import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import numpy as np
from sklearn.linear_model import LogisticRegression

from wm_base import WatermarkTask

class StudentWatermarkTask(WatermarkTask):
    def __init__(self, clean_model: nn.Module, num_models: int = 10, perturbation_factor: float = 0.01, batch_size: int = 128, device: str = 'cuda', inject_lr = 1e-4, ext_lr = 1e-3, wm_lambda = 0.01, layer_name = "layer3.1.conv2", epoch = 3, msg_bits = 128):
        """
        Initialize the student watermark task. 
        This class inherits from the base class `WatermarkTask` and implements the necessary methods.

        Args:
            clean_model (nn.Module): The clean pre-trained model.
            num_models (int): Number of perturbed models to generate.
            perturbation_factor (float): Factor to control the amount of perturbation.
            batch_size (int): Batch size for the CIFAR-10 DataLoader.
            device (str): Device to run the models on ('cpu' or 'cuda').
        """
        super().__init__(clean_model, num_models, perturbation_factor, batch_size, device)
        
        self.inject_lr = inject_lr
        self.ext_lr = ext_lr
        self.wm_lambda = wm_lambda
        self.epoch = epoch
        self.msg_bits = msg_bits
        self.wm_layer_name = layer_name
        
        # self.wm_msg = torch.rand((self.msg_bits,), dtype=torch.float32, device=self.device) *0.8+0.1
        self.wm_msg = torch.randint(0, 2, (self.msg_bits,), dtype=torch.float32, device=self.device) * 0.8 + 0.1

        wm_layer = dict(clean_model.named_modules())[self.wm_layer_name]
        flat_dim = wm_layer.weight.numel()
        
        self.w_non_list = [m.to(self.device) for m in self.clean_models]
        # self.m_r_list = [torch.rand((self.msg_bits,), dtype=torch.float32, device=self.device) * 0.8 + 0.1 for _ in self.w_non_list]
        self.m_r_list = [torch.randint(0, 2, (self.msg_bits,), dtype=torch.float32, device=self.device) * 0.8 + 0.1 for _ in self.w_non_list]

        self.extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.msg_bits),
            nn.Sigmoid()
        ).to(self.device)
        
        
    def insert_watermark(self, model: nn.Module) -> nn.Module:
        epoch = self.epoch; lambda_wm = self.wm_lambda


        watermark_model = copy.deepcopy(model).to(self.device).train()
        wm_layer = dict(watermark_model.named_modules())[self.wm_layer_name]
        wm_optimizer = optim.Adam(watermark_model.parameters(), lr=self.inject_lr, betas=(0.9, 0.999))
        ext_optimizer = optim.Adam(self.extractor.parameters(), lr=self.ext_lr, betas=(0.5, 0.999))

        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCELoss()
        
        for i in range(epoch):
            
            self.extractor.train()
            watermark_model.train()
            
            for img, labels in self.train_loader:
                img, labels = img.to(self.device), labels.to(self.device)
                
                #update extractor
                ext_epoch = 1
                for _ in range(ext_epoch):
                    ext_optimizer.zero_grad()
                    wm_w = wm_layer.weight.view(1, -1)
                    pred_msg = self.extractor(wm_w)
                    loss_pos = bce_loss(pred_msg.squeeze(), self.wm_msg)
                    loss_neg = 0
                    
                    k = min(len(self.w_non_list), 5)
                    clean_sampled = random.sample(list(zip(self.w_non_list, self.m_r_list)), k = k)
                    for w_non, m_r in clean_sampled:
                        non_layer = dict(w_non.named_modules())[self.wm_layer_name]
                        non_w = non_layer.weight.view(1, -1).detach()
                        
                        pred_mr = self.extractor(non_w)
                        loss_neg += bce_loss(pred_mr.squeeze(), m_r)
                    loss_neg /= k
                            
                    loss_wm = loss_pos + loss_neg
                    loss_wm.backward()
                    ext_optimizer.step()
                
                #update watermark model
                wm_optimizer.zero_grad()
                out = watermark_model(img)
                loss_task = ce_loss(out, labels)
                
                wm_w_new = wm_layer.weight.view(1, -1)
                pred_msg_new = self.extractor(wm_w_new.detach())
                loss_embed = bce_loss(pred_msg_new.squeeze(), self.wm_msg)
                
                lambda_curr = lambda_wm * (i + 1) / epoch
                total_loss = loss_task + lambda_curr * loss_embed
                total_loss.backward()
                wm_optimizer.step()
            
        return watermark_model.eval()

    def extract_features(self, model: nn.Module) -> torch.Tensor:
        model = model.to(self.device).eval()
        wm_layer = dict(model.named_modules())[self.wm_layer_name]
        with torch.no_grad():
            features = self.extractor(wm_layer.weight.view(1, -1)).squeeze(0)
        return features

    def train_detector(self, clean_models: list, wm_models: list, epochs: int = 100):
        
        X = []
        y = []
        
        for model in clean_models:
            feat = self.extract_features(model).cpu().numpy()
            X.append(feat)
            y.append(0)

        for model in wm_models:
            feat = self.extract_features(model).cpu().numpy()
            X.append(feat)
            y.append(1)

        X = np.vstack(X)
        y = np.array(y)

        # 訓練 sklearn logistic regression 分類器
        clf = LogisticRegression(max_iter=epochs)
        clf.fit(X, y)

        # 包裝成 PyTorch 模型（為了後續統一接口）
        class Detector(nn.Module):
            def __init__(self, sk_clf):
                super().__init__()
                self.clf = sk_clf

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x_np = x.detach().cpu().numpy()
                if x_np.ndim == 1:
                    x_np = x_np.reshape(1, -1)  # ➜ [1, 128]
                prob = self.clf.predict_proba(x_np)  # ➜ [B, 2]
                return torch.from_numpy(prob).float().to(x.device)
            
        return Detector(clf)
