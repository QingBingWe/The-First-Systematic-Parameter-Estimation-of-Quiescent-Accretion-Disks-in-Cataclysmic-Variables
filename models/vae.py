import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

LATENT_DIM= 256  # 不能换，小的不代表就好
SEQ_LENGTH = 3820
Batch_size = 32


# ----------------------------
# 1. 创建数据集
# ----------------------------
def load_spectracl_dataset(data_path):
    data = np.load(data_path)
    data = data[..., None].astype(np.float32)                  # (N, L, 1)
    ds = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(Batch_size)
    return ds



# ----------------------------
# 2. 定义变分自编码器（VAE）
# ----------------------------
class Sampling(tf.keras.layers.Layer):
    """
    重参数化技巧：从潜在分布中采样
    z = mean + epsilon * variance
    其中epsilon ~ N(0, 1)
    """
    def call(self, inputs):
        mean, logvar = inputs
        epsilon = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * epsilon

class Encoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim=LATENT_DIM, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        self.conv_layers = tf.keras.Sequential([
            tf.keras.layers.Conv1D(16, 5, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(32, 5, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(64, 5, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D()  
        ])

        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_logvar = tf.keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv_layers(inputs)
        mean = self.dense_mean(x)
        logvar = self.dense_logvar(x)
        z = self.sampling((mean, logvar))
        return z, mean, logvar


class Decoder(tf.keras.layers.Layer):
    def __init__(self, seq_length=3820, latent_dim=LATENT_DIM, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.seq_length = seq_length

        self.model = tf.keras.Sequential([
            # Dense → 小 patch
            tf.keras.layers.Dense(30 * 32, input_shape=(latent_dim,)),
            tf.keras.layers.Reshape((30, 32)),

            tf.keras.layers.Conv1DTranspose(32, 5, strides=4, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1DTranspose(16, 5, strides=4, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1DTranspose(8, 5, strides=4, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1DTranspose(4, 5, strides=2, padding="same", activation="relu"),

            tf.keras.layers.Conv1D(1, 3, padding="same", activation="tanh"),
            tf.keras.layers.Lambda(lambda x: x[:, :seq_length, :])  # ✅ 裁剪到 3820
        ])

    def call(self, inputs):
        return self.model(inputs)

class VAE(tf.keras.Model):
    """变分自编码器完整模型"""
    def __init__(self, latent_dim=LATENT_DIM, seq_length=SEQ_LENGTH, **kwargs):
        super(VAE, self).__init__(** kwargs)
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(seq_length=seq_length)
        
        # 跟踪损失
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=None):
        # 编码得到潜在变量
        z, mean, logvar = self.encoder(inputs)
        # 解码得到重构结果
        reconstruction = self.decoder(z)
        # 返回重构结果、均值和对数方差（与train_step保持一致）
        return reconstruction, mean, logvar
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            # 编码得到潜在变量
            z, mean, logvar = self.encoder(data)
            
            # 解码得到重构结果
            reconstruction = self.decoder(z)
            
            # 计算重构损失（MSE，适用于光谱这类连续数据）
            data_flat = tf.reshape(data, [tf.shape(data)[0], -1])
            reconstruction_flat = tf.reshape(reconstruction, [tf.shape(reconstruction)[0], -1])
            
            # 计算每个样本的MSE损失（保留批次维度）
            mse_per_sample = tf.reduce_mean(tf.square(data_flat - reconstruction_flat), axis=1)
            
            # 重构损失：批次内的平均损失
            reconstruction_loss = tf.reduce_mean(mse_per_sample)
            
            
            # 计算KL散度（衡量与标准正态分布的差异）
            kl_loss = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # 总损失 = 重构损失 + KL损失
            total_loss = reconstruction_loss + 3*kl_loss

        # 计算梯度并更新参数
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # 更新损失跟踪器
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# ----------------------------
# 3. 训练VAE
# ----------------------------
def train_vae(dataset, epochs=100, latent_dim=LATENT_DIM,seq_length=SEQ_LENGTH):
    # 创建保存生成样本的目录
    if not os.path.exists('../checkpoints/check2/vae'):
        os.makedirs('../checkpoints/check2/vae')
    
    # 初始化VAE模型
    vae = VAE(latent_dim=latent_dim, seq_length=seq_length)
   
        
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    
    # 训练模型
    history = vae.fit(
        dataset,
        epochs=epochs,
        verbose=1
    )
    total_params = np.sum([np.prod(v.shape) for v in vae.trainable_weights])
    print(f"Trainable params: {total_params}")
    # 绘制损失曲线
    plot_losses(history.history)
    
    return vae

# ----------------------------
# 4. 辅助函数：生成与可视化
# ----------------------------
def generate_spectra(vae,data):
    """从标准正态分布采样，生成新的光谱"""
 
    # z = tf.random.normal(shape=(num_samples, ))
    
    z = data
    generated = vae.decoder(z)
    return generated

def plot_losses(history):
    """绘制训练过程中的损失曲线"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['total_loss'])
    plt.title('total_loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['reconstruction_loss'])
    plt.title('reconsturction_loss')
    plt.xlabel('Epoch')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['kl_loss'])
    plt.title('KL_loss')
    plt.xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig('../checkpoints/check2/vae/vae_losses.png')
    plt.close()




# ----------------------------
# 5. 主函数
# ----------------------------
if __name__ == "__main__":
    # 超参数
    latent_dim = LATENT_DIM  # 潜在空间维度
    seq_length = SEQ_LENGTH
    epochs = 200     # 训练轮数
    Train = False  # 如果是生成的话把这里改成False,训练时则为True
    # 训练VAE
    if Train:
        data_path = '../data/processed_2/sdss_y.npy'
        train_dataset = load_spectracl_dataset(data_path=data_path)
        vae = train_vae(train_dataset, epochs=epochs, latent_dim=latent_dim)
        # 保存模型
        vae.save_weights('../checkpoints/check2/vae/vae_generator.h5')
        print("VAE模型已保存")
    else:
        cloudy_data = np.load("../data/cloudy_data/cloudy_dataset_256.npy")
        model_path ="../checkpoints/check2/vae/vae_generator.h5"
        vae = VAE(latent_dim=latent_dim, seq_length=seq_length)
        dummy_input = tf.zeros((1, seq_length,1))  # 1个样本，长度为3820
        _ = vae.encoder(dummy_input)  # 确保编码器创建变量
        _ = vae.decoder(tf.zeros((1, latent_dim)))  # 确保解码器创建变量
        _ = vae(dummy_input)  # 完整调用一次模型
        vae.load_weights(model_path)
        batch_size = 64
        new_cloudy_data = []

        # 将原始数据按batch_size拆分
        for i in tqdm(range(0, len(cloudy_data), batch_size),desc="生成进度"):
            # 提取当前批次（最后一批可能不足64个）
            batch_data = cloudy_data[i:i+batch_size]
            
            # 转换为numpy数组（确保是二维/三维批量格式）
            # 假设cloudy_data是列表，每个元素是单条光谱，形状为(seq_length,)
            batch_array = np.array(batch_data)  # 形状变为(batch_size, seq_length)
            
            # 一次生成整个批次（64个）
            generated_batch = generate_spectra(vae, data=batch_array)
            
            # 将生成的批次添加到结果列表
            new_cloudy_data.extend(generated_batch.numpy())  # 转换为numpy并扩展列表
        np.save("../data/generated_data_1/vae_cloudy_dataset_256.npy", new_cloudy_data)

        
        