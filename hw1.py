import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split

# ====================== 1. 数据加载与预处理模块 ======================
def load_data(data_dir='EuroSAT_RGB'):
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes = sorted(class_dirs)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    images = []
    labels = []
    print(f"正在加载 {len(classes)} 个类别的数据...")

    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        for file in os.listdir(cls_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cls_path, file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    # EuroSAT 图像为 64x64x3
                    img_arr = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_arr.flatten())  # 展平为 (12288,)
                    labels.append(class_to_idx[cls])
                except Exception as e:
                    print(f"警告：加载 {img_path} 失败: {e}")

    X = np.array(images)          # (N, 12288)
    y = np.array(labels, dtype=int)

    print(f"数据加载完成！共 {X.shape[0]} 张图像，{len(classes)} 个类别")
    print(f"类别列表: {classes}")
    return X, y, classes


# ====================== 2. 模型定义模块 (MLP + 手动反向传播) ======================
class MLP:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', weight_decay=0.0):
        """
        支持 ReLU / Sigmoid 切换
        weight_decay: L2 正则化强度（λ）
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_fn = activation
        self.weight_decay = weight_decay

        # 参数初始化（He 初始化 for ReLU，Xavier-like for Sigmoid）
        if activation == 'relu':
            scale1 = np.sqrt(2.0 / input_size)
            scale2 = np.sqrt(2.0 / hidden_size)
        else:  # sigmoid
            scale1 = np.sqrt(1.0 / input_size)
            scale2 = np.sqrt(1.0 / hidden_size)

        self.W1 = np.random.randn(input_size, hidden_size) * scale1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * scale2
        self.b2 = np.zeros((1, output_size))

    def _activation(self, z):
        if self.activation_fn == 'relu':
            return np.maximum(0, z)
        elif self.activation_fn == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        else:
            raise ValueError("仅支持 'relu' 或 'sigmoid'")

    def _activation_deriv(self, z, a):
        if self.activation_fn == 'relu':
            return (z > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            return a * (1 - a)

    def forward(self, X):
        self.X = X  # (batch, input_size)
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2

        # softmax
        exp_z2 = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        return self.probs

    def compute_loss(self, probs, y_true, include_reg=True):
        batch_size = probs.shape[0]
        log_probs = np.log(probs[np.arange(batch_size), y_true] + 1e-12)
        ce_loss = -np.mean(log_probs)

        if include_reg and self.weight_decay > 0:
            reg_loss = 0.5 * self.weight_decay * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
            return ce_loss + reg_loss
        return ce_loss

    def backward(self, y_true):
        batch_size = self.X.shape[0]

        one_hot = np.zeros((batch_size, self.output_size))
        one_hot[np.arange(batch_size), y_true] = 1.0

        # dL/dz2 (CE 梯度，已除以 batch_size)
        dz2 = (self.probs - one_hot) / batch_size

        # 输出层梯度
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # 隐藏层
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self._activation_deriv(self.z1, self.a1)

        dW1 = np.dot(self.X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # L2 正则化梯度
        if self.weight_decay > 0:
            dW1 += self.weight_decay * self.W1
            dW2 += self.weight_decay * self.W2

        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, lr):
        """SGD 参数更新"""
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def save_weights(self, filepath='best_model.npz'):
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"权重已保存至 {filepath}")

    def load_weights(self, filepath):
        """导入训练好的最优模型权重"""
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"已从 {filepath} 加载最优权重")


# ====================== 3. 训练循环模块 (Trainer) ======================
class Trainer:
    def __init__(self, model, lr=0.01, lr_decay=0.95, batch_size=128, epochs=80):
        self.model = model
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, X_train, y_train, X_val, y_val):
        """完整训练循环：SGD + LR Decay + 验证集保存最优权重"""
        n_samples = X_train.shape[0]
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

        best_val_acc = 0.0
        best_weights = None
        current_lr = self.lr

        np.random.seed(42)  

        for epoch in range(self.epochs):
            # 随机打乱
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            epoch_loss = 0.0
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]

                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                probs = self.model.forward(X_batch)
                loss = self.model.compute_loss(probs, y_batch, include_reg=True)
                epoch_loss += loss * len(batch_idx)

                # 反向传播 + 更新
                dW1, db1, dW2, db2 = self.model.backward(y_batch)
                self.model.update_params(dW1, db1, dW2, db2, current_lr)

            # 平均 train loss
            train_loss = epoch_loss / n_samples

            # 验证集评估
            val_probs = self.model.forward(X_val)
            val_loss = self.model.compute_loss(val_probs, y_val, include_reg=False)
            val_pred = np.argmax(val_probs, axis=1)
            val_acc = np.mean(val_pred == y_val)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1:3d}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

            # 保存验证集上最优权重
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = {
                    'W1': self.model.W1.copy(),
                    'b1': self.model.b1.copy(),
                    'W2': self.model.W2.copy(),
                    'b2': self.model.b2.copy()
                }

            # 学习率衰减（每 epoch 衰减）
            current_lr *= self.lr_decay

        # 恢复最优权重到模型
        if best_weights:
            self.model.W1 = best_weights['W1']
            self.model.b1 = best_weights['b1']
            self.model.W2 = best_weights['W2']
            self.model.b2 = best_weights['b2']

        print(f"训练结束！验证集最高准确率: {best_val_acc:.4f}")
        return history, best_val_acc


# ====================== 4. 超参数查找模块 ======================
def hyperparameter_search(X_train, y_train, X_val, y_val, input_size, output_size):
    """网格搜索：学习率、隐藏层大小、正则化强度"""
    lrs = [0.001, 0.01, 0.05]
    hidden_sizes = [128, 256]
    weight_decays = [0.0, 1e-4]

    results = []
    best_val_acc = -1.0
    best_combo = None

    print("开始网格搜索超参数（共 12 组组合）...")
    for lr, hidden, wd in itertools.product(lrs, hidden_sizes, weight_decays):
        print(f"测试组合: lr={lr}, hidden={hidden}, wd={wd}")
        model = MLP(input_size, hidden, output_size, activation='relu', weight_decay=wd)
        trainer = Trainer(model, lr=lr, lr_decay=0.92, batch_size=128, epochs=25)  # 搜索阶段用较少 epoch
        _, val_acc = trainer.train(X_train, y_train, X_val, y_val)

        results.append((lr, hidden, wd, val_acc))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_combo = (lr, hidden, wd)

    # 打印搜索结果表格
    print("\n" + "="*60)
    print("超参数搜索结果总结")
    print(f"{'LR':<8} {'Hidden':<8} {'WD':<10} {'Val Acc':<8}")
    print("-" * 60)
    for lr, h, wd, acc in results:
        print(f"{lr:<8} {h:<8} {wd:<10} {acc:<8.4f}")
    print("="*60)
    print(f"最佳超参数: lr={best_combo[0]}, hidden={best_combo[1]}, wd={best_combo[2]} "
          f"(Val Acc = {best_val_acc:.4f})")
    return best_combo


# ====================== 5. 测试评估 + 可视化模块 ======================
def evaluate_on_test(model, X_test, y_test, classes):
    """在独立测试集上评估：准确率 + 混淆矩阵"""
    probs = model.forward(X_test)
    pred = np.argmax(probs, axis=1)
    acc = np.mean(pred == y_test)
    print(f"测试集准确率 (Accuracy): {acc:.4f} ({acc*100:.2f}%)")

    # 混淆矩阵（纯 NumPy 实现）
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_test, pred):
        cm[t, p] += 1

    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("混淆矩阵 (Confusion Matrix):")
    header = " " * 12 + " ".join(f"{c[:10]:>10}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        line = f"{classes[i][:10]:<12} " + " ".join(f"{x:10d}" for x in row)
        print(line)

    return pred


def visualize_first_layer_weights(model, classes):
    """第一层隐藏层权重恢复成图像尺寸并可视化"""
    # W1 形状: (12288, hidden_size)，每列是一个隐藏单元的权重
    hidden = model.hidden_size
    num_show = min(16, hidden)

    plt.figure(figsize=(12, 12))
    for i in range(num_show):
        w_flat = model.W1[:, i]
        w_img = w_flat.reshape(64, 64, 3)

        # 归一化到 [0,1] 便于显示
        w_img = (w_img - w_img.min()) / (w_img.max() - w_img.min() + 1e-8)

        plt.subplot(4, 4, i + 1)
        plt.imshow(w_img)
        plt.title(f'Filter {i}')
        plt.axis('off')

    plt.suptitle('first_layer_weights', fontsize=14)
    plt.tight_layout()
    plt.savefig('first_layer_weights.png', dpi=200, bbox_inches='tight')
    plt.show()
    print("第一层权重可视化已保存为 first_layer_weights.png")

  
def error_analysis(model, X_test, y_test, pred, classes):
    """错例分析：输出前几个分类错误的样本信息"""
    errors_idx = np.where(pred != y_test)[0]

    plt.figure(figsize=(15, 6))
    for i,idx in enumerate(errors_idx[:6]):
        true_cls = classes[y_test[idx]]
        pred_cls = classes[pred[idx]]
        print(f"  样本索引 {idx:5d} | 真实: {true_cls:15s} | 预测: {pred_cls:15s}")

        img = X_test[idx].reshape(64, 64, 3)
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(f"True: {true_cls}\nPred: {pred_cls}", color='red')
        plt.axis('off')
    plt.suptitle('Error Analysis: Misclassified Samples')
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()

# ====================== 主程序 ======================
if __name__ == "__main__":
    np.random.seed(42)

    # # load ckpt
    final_model = MLP(input_size=12288, hidden_size=256, output_size=10, activation='relu', weight_decay=0.01)
    final_model.load_weights('best_model.npz')

    
    # # 1. 数据加载与预处理
    X, y, classes = load_data('EuroSAT_RGB')
    # input_size = X.shape[1]   # 64*64*3 = 12288
    # print(f"输入特征维度 (Input Size): {input_size}")
    # output_size = len(classes)

    # # 划分数据集（train:val:test ≈ 70%:10%:20%）
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train_val, y_train_val, test_size=0.125, stratify=y_train_val, random_state=42)

    # print(f"数据集划分完成 → Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # # 2. 超参数查找（网格搜索）
    # best_combo = hyperparameter_search(X_train, y_train, X_val, y_val, input_size, output_size)
    # best_lr, best_hidden, best_wd = best_combo

    # # 3. 使用最佳超参数重新训练
    # print("\n 使用最佳超参数进行最终训练（生成报告所需曲线）...")
    # final_model = MLP(input_size, best_hidden, output_size,
    #                   activation='relu', weight_decay=best_wd)
    # final_trainer = Trainer(final_model, lr=best_lr, lr_decay=0.95,
    #                         batch_size=128, epochs=200)
    # history, _ = final_trainer.train(X_train, y_train, X_val, y_val)

    # # 保存最优权重
    # final_model.save_weights('best_model.npz')

    # epochs_range = range(1, len(history['train_loss']) + 1)
    # plt.figure(figsize=(14, 5))

    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, history['train_loss'], label='Train Loss', linewidth=2)
    # plt.plot(epochs_range, history['val_loss'], label='Val Loss', linewidth=2)
    # plt.title('train/val loss curves')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, history['val_acc'], label='Val Accuracy', color='green', linewidth=2)
    # plt.title('val accuracy ')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.savefig('training_curves.png', dpi=200, bbox_inches='tight')
    # plt.show()
    # print("训练曲线已保存为 training_curves.png")

    # # 5. 测试集评估 + 混淆矩阵 + 权重可视化 + 错例分析
    pred = evaluate_on_test(final_model, X_test, y_test, classes)
    # visualize_first_layer_weights(final_model, classes)
    error_analysis(final_model, X_test, y_test, pred, classes)

