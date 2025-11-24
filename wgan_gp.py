import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
seq_length = 3820 
Batch_size = 256
latent_dim = 256


# ----------------------------
# 1. 创建数据集
# ----------------------------
def load_spectracl_dataset(data_path):
    data = np.load(data_path)
    data = data[..., None].astype(np.float32)                  # (N, L, 1)
    ds = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(Batch_size)
    return ds



# ----------------------------
# 2. 定义生成器
# ----------------------------
class Generator(tf.keras.Model):
    def __init__(self, seq_length, latent_dim=latent_dim):
        super(Generator, self).__init__()
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        
        self.model = tf.keras.Sequential([
    # 小 Dense 层：直接映射到更小的 patch
    tf.keras.layers.Dense(units=64 * 30, input_shape=(latent_dim,)),  # 只 1920 个神经元
    tf.keras.layers.Reshape((30, 64)),  # 初始长度 30
    
    # 上采样 1: 30 → 120
    tf.keras.layers.Conv1DTranspose(64, kernel_size=5, strides=4, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    
    # 上采样 2: 120 → 480
    tf.keras.layers.Conv1DTranspose(32, kernel_size=5, strides=4, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    
    # 上采样 3: 480 → 1920
    tf.keras.layers.Conv1DTranspose(16, kernel_size=5, strides=4, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    
    # 上采样 4: 1920 → 3840
    tf.keras.layers.Conv1DTranspose(8, kernel_size=5, strides=2, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    
    # 卷积精修
    tf.keras.layers.Conv1D(8, kernel_size=3, padding="same", activation="relu"),
    
    # 输出层
    tf.keras.layers.Conv1D(1, kernel_size=3, padding="same", activation="tanh"),

    # 裁剪到目标长度 3820
    tf.keras.layers.Lambda(lambda x: x[:, :3820, :])
])

    
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

# ----------------------------
# 3. 定义评论家（Critic）
# ----------------------------
class Critic(tf.keras.Model):
    def __init__(self, seq_length):
        super(Critic, self).__init__()
        self.seq_length = seq_length
        
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 5, strides=2, padding="same", input_shape=(seq_length, 1)),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(64, 5, strides=2, padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(128, 5, strides=2, padding="same"),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)  # WGAN 输出 raw score（无 sigmoid）
        ])
    
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)
# ----------------------------
# 4. 定义WGAN-GP核心：梯度惩罚
# ----------------------------
def gradient_penalty(critic, real_samples, fake_samples):
    """计算梯度惩罚，确保评论家满足Lipschitz连续性"""
    # 生成0-1之间的随机插值权重
    batch_size = real_samples.shape[0]
    epsilon = tf.random.uniform(shape=[batch_size, 1, 1], minval=0.0, maxval=1.0)
    
    # 计算真实样本和生成样本的插值
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    
    # 计算插值样本的梯度
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        critic_interpolated = critic(interpolated, training=True)
    
    # 计算梯度范数
    gradients = tape.gradient(critic_interpolated, interpolated)
    gradients = tf.reshape(gradients, (batch_size, -1))  # 展平为(batch_size, 特征数)
    gradient_norm = tf.norm(gradients, axis=1)  # 计算L2范数
    
    # 梯度惩罚 = (梯度范数 - 1)^2 的平均值
    return tf.reduce_mean(tf.square(gradient_norm - 1))

# ----------------------------
# 5. 损失函数和优化器
# ----------------------------
def generator_loss(critic_fake_output):
    """生成器损失：希望评论家对假样本打高分"""
    return -tf.reduce_mean(critic_fake_output)

def critic_loss(critic_real_output, critic_fake_output, gradient_penalty, lambda_gp=10):
    """评论家损失：包含Wasserstein损失和梯度惩罚"""
    wasserstein_loss = tf.reduce_mean(critic_fake_output) - tf.reduce_mean(critic_real_output)
    return wasserstein_loss + lambda_gp * gradient_penalty

# WGAN-GP推荐使用Adam优化器，beta1=0是关键
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0, beta_2=0.9)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0, beta_2=0.9)

# ----------------------------
# 6. 训练循环
# ----------------------------
def train_step(real_spectra, generator, critic, lambda_gp=10):
    # 生成随机潜在向量
    batch_size = real_spectra.shape[0]
    noise = tf.random.normal([batch_size, generator.latent_dim])
    
    # 训练评论家
    with tf.GradientTape() as critic_tape:
        # 生成假样本
        fake_spectra = generator(noise, training=True)
        
        # 评论家评分
        real_output = critic(real_spectra, training=True)
        fake_output = critic(fake_spectra, training=True)
        
        # 计算梯度惩罚
        gp = gradient_penalty(critic, real_spectra, fake_spectra)
        
        # 计算评论家损失
        crit_loss = critic_loss(real_output, fake_output, gp, lambda_gp)
    
    # 更新评论家权重
    critic_gradients = critic_tape.gradient(crit_loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))
    
    # 训练生成器
    with tf.GradientTape() as gen_tape:
        fake_spectra = generator(noise, training=True)
        fake_output = critic(fake_spectra, training=True)
        
        # 计算生成器损失
        gen_loss = generator_loss(fake_output)
    
    # 更新生成器权重
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    
    return gen_loss, crit_loss

def train(dataset, epochs, generator, critic, save_interval=100, critic_iterations=5, lambda_gp=10):
    # 记录损失
    gen_losses = []
    critic_losses = []
    
    # 创建保存生成样本的目录
    if not os.path.exists('../checkpoints_new/wgangp'):
        os.makedirs('../checkpoints_new/wgangp')
    
    # 训练循环
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        epoch_gen_loss = 0.0
        epoch_critic_loss = 0.0
        batch_count = 0
        
        for real_spectra in tqdm(dataset,desc="训练进度"):
            # 每轮训练多次评论家
            for _ in range(critic_iterations):
                gen_loss, critic_loss = train_step(
                    real_spectra, generator, critic, lambda_gp
                )
                epoch_critic_loss += critic_loss
            
            epoch_gen_loss += gen_loss
            batch_count += 1
        
        # 计算平均损失
        avg_gen_loss = epoch_gen_loss / batch_count
        avg_critic_loss = epoch_critic_loss / (batch_count * critic_iterations)
        gen_losses.append(avg_gen_loss.numpy())
        critic_losses.append(avg_critic_loss.numpy())
        
        # 打印每轮损失
        if (epoch + 1) % 50 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"生成器损失: {avg_gen_loss:.4f} | 评论家损失: {avg_critic_loss:.4f}")
        
        # 定期生成并保存样本
        if (epoch + 1) % save_interval == 0:
            generate_and_save_spectra(generator, epoch + 1, generator.latent_dim, seq_length)
            generator.save_weights(f'../checkpoints_new/wgangp/generator_{epoch+1}.h5')
    
    # 绘制损失曲线
    with open('../checkpoints_new/wgangp/loss.txt', "w") as f:
        for gen_loss, critic_loss in zip(gen_losses,critic_losses):
            f.write(str(gen_loss))
            f.write(',')
            f.write(str(critic_loss))
            f.write('\n')
    plot_losses(gen_losses, critic_losses)
    return generator, critic

# ----------------------------
# 7. 辅助函数：生成并可视化光谱
# ----------------------------
def generate_and_save_spectra(generator, epoch, latent_dim, seq_length):
    """生成光谱并保存可视化结果"""
    num_examples = 5
    noise = tf.random.normal([num_examples, latent_dim])
    generated_spectra = generator(noise, training=False)
    
    # 绘制生成的光谱
    plt.figure(figsize=(15, 6))
    x = np.linspace(3000, 7000, seq_length)  # 波长范围
    
    for i in range(num_examples):
        plt.subplot(1, num_examples, i + 1)
        plt.plot(x, generated_spectra[i, :, 0], color='blue')
        plt.title(f"sample {i+1}")
        plt.xlabel("wave (Å)")
        if i == 0:
            plt.ylabel("flux")
        plt.ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'../checkpoints_new/wgangp/spectrum_epoch_{epoch}.png')
    plt.close()

def plot_losses(gen_losses, critic_losses):
    """绘制生成器和评论家的损失曲线"""
    # gen_losses = [ gen_losses[i] for i in range(len(gen_losses)) if (i+1)%100 == 0]
    # critic_losses = [ critic_losses[i] for i in range(len(gen_losses)) if (i+1)%100 == 0]
    plt.figure(figsize=(10, 6))
    plt.plot(gen_losses, label='Generator Loss', color='yellow')
    plt.plot(critic_losses, label='Discriminator Loss', color='blue')
    plt.title('WGAN-GP_loss_curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('../checkpoints_new/wgangp/wgan_gp_loss_curve.png')
    plt.close()

def load_generator(weight_path):
    # 初始化生成器（结构必须与训练时一致）
    generator = Generator(seq_length)
    # 构建模型（通过输入样例触发权重创建）
    dummy_input = tf.random.normal([1, latent_dim])
    generator(dummy_input)
    # 加载权重
    generator.load_weights(weight_path)
    print(f"成功加载生成器权重：{weight_path}")
    return generator
# ----------------------------
# 8. 主函数：初始化模型并开始训练
# ----------------------------
if __name__ == "__main__":
    # 超参数
    latent_dim = 256         # 潜在向量维度
    epochs = 5000           # 训练轮数
    critic_iterations = 5     # 每轮训练中评论家的训练次数
    lambda_gp = 10            # 梯度惩罚系数（通常设为10）
    Train = False
    if Train:
        data_path = '../data/processed_2/SDSS_y.npy'
        os.makedirs("../checkpoints_new/wgangp",exist_ok=True)
        train_dataset = load_spectracl_dataset(data_path=data_path)

        # 初始化模型（3842长度）
        generator = Generator(seq_length=seq_length, latent_dim=latent_dim)
        critic = Critic(seq_length=seq_length)
        total_params = np.sum([np.prod(v.shape) for v in generator.trainable_weights])
        print(f"Trainable params: {total_params}")
        total_params = np.sum([np.prod(v.shape) for v in critic.trainable_weights])
        print(f"Trainable params: {total_params}")
        

        
        # 验证生成器输出形状
        test_noise = tf.random.normal([1, latent_dim])
        test_output = generator(test_noise)
        print(f"生成器输出形状: {test_output.shape}")  # 应显示(1, 3820, 1)
        
        # 开始训练
        trained_generator, trained_critic = train(
            train_dataset, epochs, generator, critic,
            save_interval=200, critic_iterations=critic_iterations,
            lambda_gp=lambda_gp
        )
        

        # 保存最终模型
        trained_generator.save_weights('../checkpoints_new/wgangp/generator.h5')
        trained_critic.save_weights('../checkpoints_new/wgangp/critic.h5')
        print("模型权重已保存")
    else:
        cloudy_data = np.load("../data/cloudy_data/cloudy_dataset_256.npy")
        model_path ="../checkpoints_new/wgangp/generator.h5"
        generator = load_generator(model_path)
        batch_size = 256
        new_cloudy_data = []

        # 将原始数据按batch_size拆分
        for i in tqdm(range(0, len(cloudy_data), batch_size),desc="生成进度"):
            # 提取当前批次（最后一批可能不足64个）
            batch_data = cloudy_data[i:i+batch_size]
            
            # 转换为numpy数组（确保是二维/三维批量格式）
            # 假设cloudy_data是列表，每个元素是单条光谱，形状为(seq_length,)
            batch_array = np.array(batch_data)  # 形状变为(batch_size, seq_length)
            # print(batch_array.shape)
            
            # 一次生成整个批次（64个）
            generated_spectra = generator(batch_data, training=False)
            generated_spectra = tf.clip_by_value(generated_spectra, clip_value_min=0.0, clip_value_max=1.0)
            # 将生成的批次添加到结果列表
            new_cloudy_data.extend(generated_spectra.numpy())  # 转换为numpy并扩展列表
        np.save("../data/generated_data_new/wganGP_5000_cloudy_dataset_256.npy", new_cloudy_data)
