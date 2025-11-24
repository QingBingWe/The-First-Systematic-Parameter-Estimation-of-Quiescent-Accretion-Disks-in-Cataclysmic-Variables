import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import time
from diffusers import DPMSolverMultistepScheduler
import h5py





### 1. 多样本时序数据加载器（.npy格式）
class MultiSampleSequentialDataset(Dataset):
    """加载包含多个样本的一维时序数据集
    支持的输入形状：
    - (样本数, 序列长度)：每个样本是长度为sequence_length的一维时序
    - (样本数, 序列长度, 1)：显式标明每个时刻是1维特征
    """
    def __init__(self, data_path, normalize=False, expected_seq_length=None):
        # 验证文件存在
        if not Path(data_path).exists():
            raise FileNotFoundError(f"找不到数据文件: {data_path}")
            
        # 加载.npy数据
        self.data = np.load(data_path) # 先看看cloudy前一千的效果
        print(f"原始数据形状: {self.data.shape}")
        
        # 数据形状验证与处理
        if self.data.ndim == 2:
            # 形状为 (样本数, 序列长度)，添加特征维度使其成为 (样本数, 序列长度, 1)
            self.data = self.data[..., np.newaxis]
            print(f"已转换为 (样本数, 序列长度, 1) 格式: {self.data.shape}")
        elif self.data.ndim == 3:
            if self.data.shape[2] != 1:
                raise ValueError(f"期望每个时刻1维特征，但数据有 {self.data.shape[2]} 维")
            print("数据格式正确: (样本数, 序列长度, 1)")
        else:
            raise ValueError(f"不支持 {self.data.ndim} 维数据，请提供二维或三维.npy文件")
            
        # 验证所有样本序列长度一致
        self.num_samples, self.seq_length, self.data_dim = self.data.shape
        if expected_seq_length is not None and self.seq_length != expected_seq_length:
            raise ValueError(f"期望序列长度为 {expected_seq_length}，但数据序列长度为 {self.seq_length}")
            
        self.normalize = normalize
        self.scaler = MinMaxScaler()
        
        # 数据归一化（对每个特征单独归一化）
        if self.normalize:
            # 重塑为 (num_samples*seq_length, data_dim) 进行归一化
            data_reshaped = self.data.reshape(-1, self.data_dim)
            self.data = self.scaler.fit_transform(data_reshaped).reshape(
                self.num_samples, self.seq_length, self.data_dim
            )
            print("数据已归一化到[0, 1]范围")
            
        print(f"数据集信息: 样本数={self.num_samples}, 序列长度={self.seq_length}, 特征维度={self.data_dim}")

    def __getitem__(self, idx):
        # 返回单个样本，形状保持为 (seq_length, 1)
        return torch.tensor(self.data[idx], dtype=torch.float32)

    def __len__(self):
        return self.num_samples
    
    def inverse_transform(self, data):
        """将归一化的数据转换回原始范围"""
        if self.normalize:
            # 输入数据形状应为 (batch_size, seq_length, 1)
            batch_size, seq_len, features = data.shape
            data_reshaped = data.reshape(-1, features)
            return self.scaler.inverse_transform(data_reshaped).reshape(batch_size, seq_len, features)
        return data


### 2. 扩散模型核心组件
class DiffusionModel:
    def __init__(self, seq_length, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.seq_length = seq_length
        self.data_dim = 1
        self.num_timesteps = num_timesteps
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用计算设备: {self.device}")

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self._extract(self.alphas_cumprod.sqrt(), t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract((1 - self.alphas_cumprod).sqrt(), t, x0.shape)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)

    def p_sample(self, model, xt, t):
        betas_t = self._extract(self.betas, t, xt.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract((1 - self.alphas_cumprod).sqrt(), t, xt.shape)
        sqrt_reciprocal_alphas_t = self._extract((1.0 / self.alphas).sqrt(), t, xt.shape)

        model_output = model(xt, t)
        mean = sqrt_reciprocal_alphas_t * (xt - betas_t / sqrt_one_minus_alphas_cumprod_t * model_output)

        if t[0] == 0:
            return mean
        else:
            variance = betas_t
            noise = torch.randn_like(xt)
            return mean + torch.sqrt(variance) * noise

    def p_sample_loop(self, model, batch_size=16):
        model.eval()
        with torch.no_grad():
            xt = torch.randn((batch_size, self.seq_length, self.data_dim), device=self.device)
            for i in reversed(range(0, self.num_timesteps)):
                t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
                xt = self.p_sample(model, xt, t)
        model.train()
        return torch.clamp(xt, 0.0, 1.0)

    def dpm_sample_loop(self, model, xt, steps=25, batch_size=64, use_fp16=True):
        """
        使用 DPMSolverMultistepScheduler 进行高效采样。
        - model: 你的 TemporalDenoiseModel，输入形状 (B, L, 1), 返回 (B, L, 1)
        - xt: 初始噪声（或来自数据的初始张量），形状 (B, L, 1)
        - steps: 采样步数 (比如 25/50/100)
        - batch_size: 当前批次大小
        - use_fp16: 是否在推理中使用混合精度（autocast）；如果显存充足且追求稳定性，可设为 False
        """
        model.eval()
        # scheduler 初始化（与训练时 num_timesteps 对齐）
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=self.num_timesteps)
        scheduler.set_timesteps(steps)

        # diffusers 通常期望形状 [B, C, L]，我们把 (B, L, 1) -> (B, 1, L)
        xt = xt.to(self.device)
        xt = xt.permute(0, 2, 1).contiguous()  # (B, 1, L)

        # 选择混合精度上下文（只影响运算，不改变 model 权重 dtype）
        amp_ctx = torch.cuda.amp.autocast if (use_fp16 and torch.cuda.is_available()) else torch.no_grad

        with torch.no_grad():
            # 使用 autocast 作为上下文（若可用），否则 noop
            if use_fp16 and torch.cuda.is_available():
                amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
            else:
                amp_ctx = torch.no_grad()

            with amp_ctx:
                for t in scheduler.timesteps:
                    # scheduler.timesteps 的元素可能是 float/np.int -> 把它转换成 int timestep 传给 model
                    t_int = int(float(t))
                    t_batch = torch.full((batch_size,), t_int, device=self.device, dtype=torch.long)

                    # model 需要 (B, L, 1)
                    model_in = xt.permute(0, 2, 1)  # -> (B, L, 1)
                    noise_pred = model(model_in, t_batch)  # -> (B, L, 1)
                    noise_pred = noise_pred.permute(0, 2, 1)  # -> (B, 1, L)

                    # scheduler.step 接受 noise_pred, timestep, sample (B, C, L)
                    out = scheduler.step(noise_pred, t, xt)
                    # out 里面通常有 prev_sample 属性
                    xt = out.prev_sample

        # 转回 (B, L, 1)，转为 float32 并 clamp
        xt = xt.permute(0, 2, 1).float().to(self.device)
        return torch.clamp(xt, 0.0, 1.0)
    def cloudy_sample_loop(self,model,batch_size=16):
        model.eval()
        with torch.no_grad():
            # 初始化为纯噪声，形状: (batch_size, seq_length, 1)
            xt = torch.randn((batch_size, self.seq_length, self.data_dim), device=self.device)
            
            for i in reversed(range(0, self.num_timesteps)):
                t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
                xt = self.p_sample(model, xt, t)
        model.train()
        # 裁剪到[0,1]范围
        return torch.clamp(xt, 0.0, 1.0)


### 3. 时序去噪网络
class TimeEmbedding(nn.Module):
    """时间步嵌入模块：适配时序数据"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.emb_base = math.log(10000) / (self.half_dim - 1)

    def forward(self, t):
        device = t.device
        # 根据输入设备动态创建嵌入
        emb = torch.exp(torch.arange(self.half_dim, device=device) * -self.emb_base)
        t = t.unsqueeze(1).float()
        emb = t * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb


class TransformerEncoderLayer(nn.Module):
    """改进的Transformer编码器层，增强时序特征捕捉"""
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()  # 使用GELU激活函数，更适合时序数据

    def forward(self, x):
        # 自注意力机制
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x


class TemporalDenoiseModel(nn.Module):
    """一维时序数据的去噪模型（每个时刻1维）"""
    def __init__(self, seq_length, d_model=64, nhead=2, num_layers=3):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model  # Transformer特征维度
        self.time_emb_dim = d_model  # 时间嵌入维度与模型维度一致
        
        # 时间嵌入模块
        self.time_embedding = TimeEmbedding(self.time_emb_dim)
        
        # 输入投影：将1维特征映射到d_model维度
        self.input_proj = nn.Linear(1, d_model)
        
        # 位置编码：时序数据需要位置信息
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_length, d_model))
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead) 
            for _ in range(num_layers)
        ])
        
        # 时间嵌入投影：将时间特征融入时序特征
        self.time_proj = nn.Linear(self.time_emb_dim, d_model)
        
        # 输出投影：将d_model维度映射回1维特征
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x, t):
        # x形状: (batch_size, seq_length, 1)
        batch_size = x.shape[0]
        
        # 生成时间嵌入并投影
        t_emb = self.time_embedding(t)  # 形状: (batch_size, time_emb_dim)
        t_emb = self.time_proj(t_emb)   # 形状: (batch_size, d_model)
        t_emb = t_emb.unsqueeze(1)      # 形状: (batch_size, 1, d_model)，用于广播
        
        # 输入投影并添加位置编码
        x = self.input_proj(x)  # 形状: (batch_size, seq_length, d_model)
        x = x + self.pos_encoder  # 添加位置编码
        x = x + t_emb  # 添加时间嵌入
        
        # Transformer编码器处理
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 输出投影：回到1维特征
        x = self.output_proj(x)  # 形状: (batch_size, seq_length, 1)
        
        return x


### 4. 模型训练与可视化函数
def train(diffusion, model, train_loader, epochs=50, lr=3e-4, save_path="temporal_diffusion_model.pth"):
    """训练时序扩散模型"""
    device = diffusion.device
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # 噪声预测损失（MSE）
    model.train()
    
    # 创建样本保存目录
    os.makedirs(os.path.join(save_path,"samples"), exist_ok=True)
    losses = []
    samllest_loss=1e9
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # 确保输入形状正确: (batch_size, seq_length, 1)
            batch = batch.to(device)
            batch_size = batch.shape[0]
            
            # 随机采样时间步
            t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device).long()
            # 生成带噪声的样本
            noise = torch.randn_like(batch)  # 噪声形状与batch一致: (batch_size, seq_length, 1)
            x_t = diffusion.q_sample(x0=batch, t=t, noise=noise)
            # 模型预测噪声
            noise_pred = model(x_t, t)  # 预测噪声形状应与noise一致
            
            # 计算损失并优化
            loss = criterion(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(train_loader)
        if avg_loss<samllest_loss:
            samllest_loss=avg_loss
            torch.save(model.state_dict(), os.path.join(save_path,'best_model.pth'))
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
        losses.append(avg_loss)
        # 每10个epoch保存一次生成样本
        if (epoch + 1) % 10 == 0:
            samples = diffusion.p_sample_loop(model, batch_size=4)
            save_samples(samples, os.path.join(save_path,f"samples/samples_epoch_{epoch+1}.png"))
    plt.figure(figsize=(12,9))
    plt.plot(list(range(len(losses))),losses)
    plt.savefig(os.path.join(save_path,'loss.png'),dpi=300)
    plt.close()
    with open(os.path.join(save_path,"loss.txt"),"w") as f:
        for i in losses:
            f.write(str(i))
            f.write('\n')
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_path,"temporal_diffusion_model.pth"))
    print(f"模型训练完成并保存到 {save_path}")


def save_samples(samples, filename):
    """保存时序样本可视化结果"""
    samples = samples.cpu().numpy()
    plt.figure(figsize=(12, 8))
    
    # 绘制每个生成的时序样本
    for i in range(min(4, samples.shape[0])):
        plt.subplot(2, 2, i+1)
        # 提取时序序列: (seq_length, 1) -> (seq_length,)
        seq = samples[i, :, 0]
        plt.plot(seq)
        plt.title(f" {i+1}")
        plt.xlabel("time")
        plt.ylabel("value")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()



### 5. 主函数：运行时序扩散模型
def main():
    Train = False
    BATCH_SIZE = 16  # transformer占用内存很大，所以batch_size尽量小
    EPOCHS = 200
    NUM_TIMESTEPS = 1000
    EXPECTED_SEQ_LENGTH = 3820  
    # 初始化扩散模型和时序去噪网络
    diffusion = DiffusionModel(
        seq_length=EXPECTED_SEQ_LENGTH,  # 从数据集中获取序列长度
        num_timesteps=NUM_TIMESTEPS
    )
    
    model = TemporalDenoiseModel(
        seq_length=EXPECTED_SEQ_LENGTH,
        d_model=64,  # 模型特征维度
        nhead=2,     # 注意力头数
        num_layers=3 # Transformer层数
    )
   

    # 统计可训练参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
     # 推理加速选项
    use_fp16 = True      
    use_compile = True  
    
    # 训练模型
    if Train:
        DATA_PATH = "../data/processed_2/sdss_y.npy"
        train_dataset = MultiSampleSequentialDataset(
                data_path=DATA_PATH,
                normalize=False,
                expected_seq_length=EXPECTED_SEQ_LENGTH
                )
   
        train_loader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True, 
                                  num_workers=4
                                  )
        save_dir = "../checkpoints/check2/diffusion"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        train(diffusion, model, train_loader, epochs=EPOCHS, save_path=save_dir)
    else:
        batch_size=64
        cloudy_data_path = "../data/cloudy_data/cloudy_dataset.npy"
        cloudy_dataset = MultiSampleSequentialDataset(
            data_path=cloudy_data_path,
            normalize=False,
            expected_seq_length=EXPECTED_SEQ_LENGTH
            )

        cloudy_loader = DataLoader(cloudy_dataset, 
                                   batch_size=batch_size, 
                                   shuffle=False, 
                                   num_workers=0
                                   )
        model.load_state_dict(torch.load('../checkpoints/check2/diffusion/temporal_diffusion_model.pth', map_location='cpu'))
        model.to(diffusion.device)


        if use_fp16 and torch.cuda.is_available():
        
            model.half()
            for module in model.modules():
                if isinstance(module, nn.LayerNorm):
                    module.float()

    
        if use_compile:
            try:
                model = torch.compile(model)
                print("torch.compile 成功启用")
            except Exception as e:
                print("torch.compile 未能启用，继续使用原始模型：", e)

        model.to(diffusion.device)
        pbar = tqdm(cloudy_loader)
        device = diffusion.device
        file_name = '../data/generated_data_1/cloudy_dataset_diffusion_best.h5'
        with h5py.File(file_name,"w") as f:
            dset = f.create_dataset("data", (len(cloudy_dataset), 3820), dtype='float32')  
            idx = 0
            for batch in pbar:
                # 确保输入形状正确: (batch_size, seq_length, 1)
                batch = batch.to(device)
                batch_size = batch.shape[0]
                xt = batch
                xt = diffusion.dpm_sample_loop(model, xt,steps=45,batch_size=batch_size)
                x_data = xt.cpu().numpy()
                dset[idx:idx+batch_size] = x_data.squeeze()
                idx += batch_size
           

    
if __name__ == "__main__":
    main()
    