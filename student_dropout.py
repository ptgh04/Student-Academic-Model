"""
=====================================================================
🎓 DỰ ĐOÁN DROPOUT & ACADEMIC SUCCESS - FULL IMPLEMENTATION
Tự implement: ID3 Decision Tree + Naive Bayes
Chạy trên máy local
=====================================================================
"""

import pandas as pd
import numpy as np
from collections import Counter
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report


# ==================== 1. CÂY QUYẾT ĐỊNH ID3 ====================
class ID3DecisionTree:
    """Thuật toán ID3 Decision Tree - Tự implement hoàn toàn"""

    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = None

    def entropy(self, y):
        """
        Tính Entropy của tập dữ liệu
        H(S) = -Σ p(c) * log2(p(c))
        """
        if len(y) == 0:
            return 0

        counter = Counter(y)
        entropy_val = 0.0
        total = len(y)

        for count in counter.values():
            if count == 0:
                continue
            prob = count / total
            entropy_val -= prob * math.log2(prob)

        return entropy_val

    def information_gain(self, X_col, y, threshold=None):
        """
        Tính Information Gain
        IG(S, A) = H(S) - Σ |Sv|/|S| * H(Sv)
        """
        parent_entropy = self.entropy(y)
        n = len(y)

        if threshold is not None:
            # Biến liên tục: chia theo threshold
            left_mask = X_col <= threshold
            right_mask = X_col > threshold

            n_left = sum(left_mask)
            n_right = sum(right_mask)

            if n_left == 0 or n_right == 0:
                return 0.0

            left_entropy = self.entropy(y[left_mask])
            right_entropy = self.entropy(y[right_mask])

            weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
        else:
            # Biến rời rạc: chia theo từng giá trị
            values = np.unique(X_col)
            weighted_entropy = 0.0

            for value in values:
                mask = X_col == value
                n_subset = sum(mask)

                if n_subset == 0:
                    continue

                subset_entropy = self.entropy(y[mask])
                weighted_entropy += (n_subset / n) * subset_entropy

        return parent_entropy - weighted_entropy

    def find_best_split(self, X, y, feature_idx):
        """Tìm điểm chia tốt nhất cho một feature"""
        X_col = X[:, feature_idx]
        unique_values = np.unique(X_col)

        # Quyết định biến liên tục hay rời rạc
        if len(unique_values) <= 10:
            # Biến rời rạc
            ig = self.information_gain(X_col, y)
            return ig, None
        else:
            # Biến liên tục: tìm threshold tốt nhất
            sorted_values = np.sort(unique_values)
            candidate_thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

            # Sample nếu có quá nhiều thresholds
            if len(candidate_thresholds) > 30:
                candidate_thresholds = np.random.choice(
                    candidate_thresholds, 30, replace=False
                )

            best_ig = 0.0
            best_threshold = None

            for threshold in candidate_thresholds:
                ig = self.information_gain(X_col, y, threshold)
                if ig > best_ig:
                    best_ig = ig
                    best_threshold = threshold

            return best_ig, best_threshold

    def build_tree(self, X, y, depth=0):
        """Xây dựng cây quyết định theo thuật toán ID3"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Điều kiện dừng
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                n_classes == 1):
            # Tạo leaf node với class phổ biến nhất
            leaf_value = Counter(y).most_common(1)[0][0]
            return {
                'leaf': True,
                'value': leaf_value,
                'samples': n_samples
            }

        # Tìm feature tốt nhất để split
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            gain, threshold = self.find_best_split(X, y, feature_idx)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

        # Nếu không tìm được split tốt
        if best_gain == 0.0 or best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {
                'leaf': True,
                'value': leaf_value,
                'samples': n_samples
            }

        # Thực hiện split
        X_col = X[:, best_feature]

        if best_threshold is not None:
            # Split theo threshold (biến liên tục)
            left_mask = X_col <= best_threshold
            right_mask = X_col > best_threshold

            if sum(left_mask) == 0 or sum(right_mask) == 0:
                leaf_value = Counter(y).most_common(1)[0][0]
                return {
                    'leaf': True,
                    'value': leaf_value,
                    'samples': n_samples
                }

            left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
            right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

            return {
                'leaf': False,
                'feature': best_feature,
                'threshold': best_threshold,
                'left': left_subtree,
                'right': right_subtree,
                'samples': n_samples,
                'gain': best_gain
            }
        else:
            # Split theo giá trị rời rạc
            unique_values = np.unique(X_col)
            branches = {}

            for value in unique_values:
                mask = X_col == value
                n_subset = sum(mask)

                if n_subset >= self.min_samples_split:
                    branches[value] = self.build_tree(
                        X[mask], y[mask], depth + 1
                    )

            if len(branches) == 0:
                leaf_value = Counter(y).most_common(1)[0][0]
                return {
                    'leaf': True,
                    'value': leaf_value,
                    'samples': n_samples
                }

            # Lưu default class cho giá trị chưa gặp
            default_class = Counter(y).most_common(1)[0][0]

            return {
                'leaf': False,
                'feature': best_feature,
                'threshold': None,
                'branches': branches,
                'default_class': default_class,
                'samples': n_samples,
                'gain': best_gain
            }

    def fit(self, X, y, feature_names=None):
        """Huấn luyện mô hình"""
        print("🌳 Đang xây dựng cây quyết định ID3...")
        self.feature_names = feature_names
        self.tree = self.build_tree(X, y)
        print("✅ Đã xây dựng xong cây ID3!")
        return self

    def predict_sample(self, x, tree):
        """Dự đoán cho một mẫu"""
        if tree['leaf']:
            return tree['value']

        feature_val = x[tree['feature']]

        if tree['threshold'] is not None:
            # Continuous feature
            if feature_val <= tree['threshold']:
                return self.predict_sample(x, tree['left'])
            else:
                return self.predict_sample(x, tree['right'])
        else:
            # Categorical feature
            if feature_val in tree['branches']:
                return self.predict_sample(x, tree['branches'][feature_val])
            else:
                # Giá trị chưa gặp: dùng default class
                return tree.get('default_class', 0)

    def predict(self, X):
        """Dự đoán cho tập dữ liệu"""
        predictions = []
        for x in X:
            pred = self.predict_sample(x, self.tree)
            predictions.append(pred)
        return np.array(predictions)

    def print_tree(self, tree=None, depth=0, prefix="Root"):
        """In cấu trúc cây (để debug)"""
        if tree is None:
            tree = self.tree

        indent = "  " * depth

        if tree['leaf']:
            print(f"{indent}{prefix}: Leaf -> Class {tree['value']} (samples: {tree['samples']})")
        else:
            feature_name = (self.feature_names[tree['feature']]
                            if self.feature_names else f"Feature {tree['feature']}")

            if tree['threshold'] is not None:
                print(f"{indent}{prefix}: {feature_name} <= {tree['threshold']:.2f} "
                      f"(gain: {tree['gain']:.4f}, samples: {tree['samples']})")
                self.print_tree(tree['left'], depth + 1, "Left")
                self.print_tree(tree['right'], depth + 1, "Right")
            else:
                print(f"{indent}{prefix}: {feature_name} (gain: {tree['gain']:.4f}, "
                      f"samples: {tree['samples']})")
                for value, subtree in tree['branches'].items():
                    self.print_tree(subtree, depth + 1, f"Value={value}")


# ==================== 2. NAIVE BAYES ====================
class NaiveBayes:
    """Gaussian Naive Bayes - Tự implement hoàn toàn"""

    def __init__(self):
        self.classes = None
        self.class_priors = {}  # P(C)
        self.means = {}  # μ cho mỗi feature và class
        self.variances = {}  # σ² cho mỗi feature và class
        self.epsilon = 1e-9  # Tránh chia cho 0

    def fit(self, X, y):
        """
        Huấn luyện Naive Bayes
        Tính P(C), mean và variance cho mỗi class
        """
        print("🧮 Đang huấn luyện Naive Bayes...")

        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        for c in self.classes:
            # Lấy tất cả samples thuộc class c
            X_c = X[y == c]

            # Tính prior probability: P(C)
            self.class_priors[c] = len(X_c) / n_samples

            # Tính mean và variance cho mỗi feature
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + self.epsilon

        print("✅ Đã huấn luyện xong Naive Bayes!")
        return self

    def gaussian_probability(self, x, mean, var):
        """
        Tính xác suất theo phân phối Gaussian
        P(x|μ,σ²) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
        """
        coefficient = 1.0 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coefficient * exponent

    def predict_sample(self, x):
        """
        Dự đoán cho một mẫu
        Áp dụng Bayes' theorem:
        P(C|X) ∝ P(C) * Π P(Xi|C)
        """
        posteriors = {}

        for c in self.classes:
            # Log probability để tránh underflow
            log_prior = np.log(self.class_priors[c])

            # Tính log likelihood cho tất cả features
            log_likelihood = 0.0
            for i in range(len(x)):
                prob = self.gaussian_probability(
                    x[i],
                    self.means[c][i],
                    self.variances[c][i]
                )
                # Tránh log(0)
                log_likelihood += np.log(prob + self.epsilon)

            # Log posterior
            posteriors[c] = log_prior + log_likelihood

        # Trả về class có posterior cao nhất
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        """Dự đoán cho tập dữ liệu"""
        predictions = []
        for x in X:
            pred = self.predict_sample(x)
            predictions.append(pred)
        return np.array(predictions)

    def predict_proba(self, X):
        """
        Tính xác suất dự đoán cho từng class
        Sử dụng softmax để normalize
        """
        probabilities = []

        for x in X:
            posteriors = {}

            for c in self.classes:
                log_prior = np.log(self.class_priors[c])
                log_likelihood = 0.0

                for i in range(len(x)):
                    prob = self.gaussian_probability(
                        x[i],
                        self.means[c][i],
                        self.variances[c][i]
                    )
                    log_likelihood += np.log(prob + self.epsilon)

                posteriors[c] = log_prior + log_likelihood

            # Normalize bằng softmax
            max_log_prob = max(posteriors.values())
            exp_probs = {c: np.exp(log_prob - max_log_prob)
                         for c, log_prob in posteriors.items()}
            total = sum(exp_probs.values())

            # Tạo array xác suất theo thứ tự classes
            probs = np.array([exp_probs[c] / total for c in self.classes])
            probabilities.append(probs)

        return np.array(probabilities)


# ==================== 3. XỬ LÝ DỮ LIỆU ====================
def load_and_preprocess_data(file_path):
    """Đọc và tiền xử lý dữ liệu"""
    print(f"\n{'=' * 70}")
    print("📊 XỬ LÝ DỮ LIỆU")
    print(f"{'=' * 70}\n")

    print(f"📁 Đang đọc file: {file_path}")
    df = pd.read_csv(file_path)

    print(f"✅ Đã đọc {len(df)} mẫu với {len(df.columns)} cột")

    # Hiển thị thông tin
    print(f"\n📋 Các cột trong dataset:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        print(f"   {i:2d}. {col:30s} - Type: {dtype}, Unique: {n_unique}")

    # Xử lý missing values
    print(f"\n🔧 Xử lý dữ liệu thiếu...")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"   Tìm thấy {missing_counts.sum()} giá trị thiếu")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print(f"   ✓ Không có dữ liệu thiếu")

    # Tìm cột target
    possible_targets = ['Target', 'target', 'label', 'class', 'output']
    target_col = None

    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        target_col = df.columns[-1]

    print(f"\n🎯 Cột target: {target_col}")

    # Tách features và target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Lưu tên features
    feature_names = X.columns.tolist()

    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded: {col}")

    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    print(f"\n📊 Phân bố target:")
    for i, class_name in enumerate(le_target.classes_):
        count = sum(y_encoded == i)
        print(f"   {class_name:20s}: {count:5d} ({count / len(y_encoded) * 100:.2f}%)")

    return X.values, y_encoded, le_target.classes_, feature_names


# ==================== 4. ĐÁNH GIÁ ====================
def calculate_accuracy(y_true, y_pred):
    """Tính accuracy"""
    correct = sum(y_true == y_pred)
    total = len(y_true)
    return correct / total


def calculate_metrics(y_true, y_pred, classes):
    """Tính precision, recall, f1-score cho từng class"""
    metrics = {}

    for i, class_name in enumerate(classes):
        # True Positives, False Positives, False Negatives
        tp = sum((y_true == i) & (y_pred == i))
        fp = sum((y_true != i) & (y_pred == i))
        fn = sum((y_true == i) & (y_pred != i))

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)

        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': sum(y_true == i)
        }

    return metrics


def evaluate_model(y_true, y_pred, model_name, classes):
    """Đánh giá chi tiết mô hình"""
    print(f"\n{'=' * 70}")
    print(f"📊 KẾT QUẢ MÔ HÌNH: {model_name}")
    print(f"{'=' * 70}")

    # Accuracy
    accuracy = calculate_accuracy(y_true, y_pred)
    print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Metrics cho từng class
    metrics = calculate_metrics(y_true, y_pred, classes)

    print(f"\n📈 Chi tiết từng class:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)

    for class_name, metric in metrics.items():
        print(f"{class_name:<15} "
              f"{metric['precision']:>10.4f} "
              f"{metric['recall']:>10.4f} "
              f"{metric['f1-score']:>10.4f} "
              f"{metric['support']:>10d}")

    # Macro average
    macro_precision = np.mean([m['precision'] for m in metrics.values()])
    macro_recall = np.mean([m['recall'] for m in metrics.values()])
    macro_f1 = np.mean([m['f1-score'] for m in metrics.values()])

    print("-" * 70)
    print(f"{'Macro Avg':<15} "
          f"{macro_precision:>10.4f} "
          f"{macro_recall:>10.4f} "
          f"{macro_f1:>10.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(cm)

    return accuracy, cm


def plot_comparison(accuracies, cms, classes):
    """Vẽ biểu đồ so sánh"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. So sánh accuracy
    models = list(accuracies.keys())
    accs = list(accuracies.values())

    colors = ['#3498db', '#e74c3c']
    bars = axes[0].bar(models, accs, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('So sánh Độ chính xác', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    for i, (bar, v) in enumerate(zip(bars, accs)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{v:.4f}\n({v * 100:.2f}%)',
                     ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 2-3. Confusion matrices
    for idx, (model_name, cm) in enumerate(cms.items(), start=1):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    ax=axes[idx], cbar_kws={'label': 'Số mẫu'},
                    linewidths=0.5, linecolor='gray')
        axes[idx].set_title(f'Confusion Matrix - {model_name}',
                            fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Nhãn thực tế', fontweight='bold')
        axes[idx].set_xlabel('Nhãn dự đoán', fontweight='bold')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Đã lưu biểu đồ: model_comparison.png")
    plt.show()


# ==================== 5. DỰ ĐOÁN DỮ LIỆU MỚI ====================
def predict_new_sample(id3_model, nb_model, classes, feature_names, X_sample):
    """Dự đoán mẫu mới"""
    print(f"\n{'=' * 70}")
    print("🔮 KẾT QUẢ DỰ ĐOÁN")
    print(f"{'=' * 70}")

    # ID3 prediction
    pred_id3 = id3_model.predict(X_sample)[0]
    print(f"\n🌳 ID3 Decision Tree: {classes[pred_id3]}")

    # Naive Bayes prediction
    pred_nb = nb_model.predict(X_sample)[0]
    pred_proba = nb_model.predict_proba(X_sample)[0]
    print(f"🧮 Naive Bayes: {classes[pred_nb]}")

    # Xác suất
    print(f"\n📊 Xác suất dự đoán (Naive Bayes):")
    for i, class_name in enumerate(classes):
        bar = '█' * int(pred_proba[i] * 50)
        print(f"   {class_name:20s} {bar} {pred_proba[i] * 100:.2f}%")

    # Kết luận
    if pred_id3 == pred_nb:
        print(f"\n✅ CẢ HAI MÔ HÌNH ĐỒNG Ý: {classes[pred_id3]}")
    else:
        print(f"\n⚠️  HAI MÔ HÌNH KHÁC NHAU:")
        print(f"   - ID3: {classes[pred_id3]}")
        print(f"   - Naive Bayes: {classes[pred_nb]}")

    return pred_id3, pred_nb


# ==================== MAIN ====================
def main():
    print("=" * 70)
    print("🎓 DỰ ĐOÁN DROPOUT & ACADEMIC SUCCESS CỦA HỌC SINH")
    print("=" * 70)
    print("🤖 Tự implement: ID3 Decision Tree + Naive Bayes")
    print("💻 Chạy trên máy local")
    print("=" * 70)

    # 1. Đọc dữ liệu
    file_path = input("\n📁 Nhập đường dẫn file CSV: ").strip()
    if not file_path:
        file_path = 'dataset.csv'

    try:
        X, y, classes, feature_names = load_and_preprocess_data(file_path)
    except Exception as e:
        print(f"\n❌ Lỗi đọc file: {e}")
        print("💡 Đảm bảo file CSV tồn tại và có định dạng đúng")
        return

    # 2. Chia train-test
    print(f"\n{'=' * 70}")
    print("🔄 CHIA DỮ LIỆU TRAIN/TEST")
    print(f"{'=' * 70}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n✅ Kết quả:")
    print(f"   - Train: {len(X_train)} mẫu ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"   - Test:  {len(X_test)} mẫu ({len(X_test) / len(X) * 100:.1f}%)")

    # 3. Huấn luyện ID3
    print(f"\n{'=' * 70}")
    print("🌳 HUẤN LUYỆN ID3 DECISION TREE")
    print(f"{'=' * 70}")

    id3_model = ID3DecisionTree(max_depth=10, min_samples_split=5)
    id3_model.fit(X_train, y_train, feature_names)

    print("\n🔮 Dự đoán trên tập test...")
    y_pred_id3 = id3_model.predict(X_test)
    acc_id3, cm_id3 = evaluate_model(y_test, y_pred_id3, "ID3 Decision Tree", classes)

    # 4. Huấn luyện Naive Bayes
    print(f"\n{'=' * 70}")
    print("🧮 HUẤN LUYỆN NAIVE BAYES")
    print(f"{'=' * 70}")

    nb_model = NaiveBayes()
    nb_model.fit(X_train, y_train)

    print("\n🔮 Dự đoán trên tập test...")
    y_pred_nb = nb_model.predict(X_test)
    acc_nb, cm_nb = evaluate_model(y_test, y_pred_nb, "Naive Bayes", classes)

    # 5. So sánh
    print(f"\n{'=' * 70}")
    print("📊 TỔNG KẾT SO SÁNH")
    print(f"{'=' * 70}")
    print(f"\n🌳 ID3 Decision Tree: {acc_id3:.4f} ({acc_id3 * 100:.2f}%)")
    print(f"🧮 Naive Bayes:       {acc_nb:.4f} ({acc_nb * 100:.2f}%)")

    diff = abs(acc_id3 - acc_nb)
    if acc_id3 > acc_nb:
        print(f"\n🏆 ID3 Decision Tree tốt hơn {diff * 100:.2f}%")
    elif acc_nb > acc_id3:
        print(f"\n🏆 Naive Bayes tốt hơn {diff * 100:.2f}%")
    else:
        print(f"\n🤝 Hai mô hình có độ chính xác tương đương")

    # 6. Vẽ biểu đồ
    print(f"\n{'=' * 70}")
    print("📊 TẠO BIỂU ĐỒ SO SÁNH")
    print(f"{'=' * 70}")

    accuracies = {'ID3 Decision Tree': acc_id3, 'Naive Bayes': acc_nb}
    cms = {'ID3 Decision Tree': cm_id3, 'Naive Bayes': cm_nb}
    plot_comparison(accuracies, cms, classes)

    # 7. Dự đoán mẫu mới
    print(f"\n{'=' * 70}")
    print("🔮 DỰ ĐOÁN DỮ LIỆU MỚI")
    print(f"{'=' * 70}")

    while True:
        print("\n📌 Chọn cách dự đoán:")
        print("   1. Nhập thủ công")
        print("   2. Dùng mẫu random từ dataset")
        print("   3. In cấu trúc cây ID3 (để hiểu thuật toán)")
        print("   4. Thoát")

        choice = input("\n👉 Lựa chọn (1/2/3/4): ").strip()

        if choice == '4':
            break

        elif choice == '3':
            # In cấu trúc cây
            print(f"\n{'=' * 70}")
            print("🌳 CẤU TRÚC CÂY QUYẾT ĐỊNH ID3")
            print(f"{'=' * 70}\n")

            depth_limit = input("Độ sâu tối đa để hiển thị (Enter = 3): ").strip()
            depth_limit = int(depth_limit) if depth_limit else 3

            def print_tree_limited(tree, depth=0, prefix="Root", max_depth=3):
                if depth >= max_depth:
                    return

                indent = "  " * depth

                if tree['leaf']:
                    print(f"{indent}{prefix}: 🍃 Leaf → Class {classes[tree['value']]} "
                          f"(samples: {tree['samples']})")
                else:
                    feature_name = (feature_names[tree['feature']]
                                    if feature_names else f"Feature {tree['feature']}")

                    if tree['threshold'] is not None:
                        print(f"{indent}{prefix}: 📊 {feature_name} <= {tree['threshold']:.2f} "
                              f"(IG: {tree['gain']:.4f}, samples: {tree['samples']})")
                        print_tree_limited(tree['left'], depth + 1, "├─ Left ", max_depth)
                        print_tree_limited(tree['right'], depth + 1, "└─ Right", max_depth)
                    else:
                        print(f"{indent}{prefix}: 📊 {feature_name} "
                              f"(IG: {tree['gain']:.4f}, samples: {tree['samples']})")
                        branches = list(tree['branches'].items())
                        for i, (value, subtree) in enumerate(branches):
                            if i < len(branches) - 1:
                                print_tree_limited(subtree, depth + 1, f"├─ Val={value}", max_depth)
                            else:
                                print_tree_limited(subtree, depth + 1, f"└─ Val={value}", max_depth)

            print_tree_limited(id3_model.tree, max_depth=depth_limit)
            continue

        elif choice == '2':
            # Random sample
            random_idx = np.random.randint(0, len(X))
            X_sample = X[random_idx:random_idx + 1]

            print(f"\n✅ Mẫu ngẫu nhiên (index: {random_idx}):")
            print(f"\n📊 Giá trị các đặc trưng:")
            for i, (fname, val) in enumerate(zip(feature_names, X_sample[0])):
                if i < 10:  # Chỉ hiển thị 10 features đầu
                    print(f"   {i + 1:2d}. {fname:30s} = {val:.2f}")

            if len(feature_names) > 10:
                print(f"   ... và {len(feature_names) - 10} features khác")

            predict_new_sample(id3_model, nb_model, classes, feature_names, X_sample)

        elif choice == '1':
            # Manual input
            print(f"\n📝 Nhập dữ liệu cho {len(feature_names)} đặc trưng")
            print("💡 Nhấn Enter để dùng giá trị trung bình")
            print("💡 Nhập 'skip' để bỏ qua các features còn lại\n")

            feature_means = X.mean(axis=0)
            new_input = []
            skip_rest = False

            for i, (fname, mean_val) in enumerate(zip(feature_names, feature_means)):
                if skip_rest:
                    new_input.append(mean_val)
                    continue

                user_input = input(f"  {i + 1:2d}/{len(feature_names)}. {fname} "
                                   f"(mean: {mean_val:.2f}): ").strip()

                if user_input.lower() == 'skip':
                    print(f"     ⏩ Dùng giá trị trung bình cho các features còn lại")
                    new_input.append(mean_val)
                    skip_rest = True
                elif user_input == "":
                    new_input.append(mean_val)
                else:
                    try:
                        new_input.append(float(user_input))
                    except:
                        print(f"     ⚠️  Lỗi, dùng mean: {mean_val:.2f}")
                        new_input.append(mean_val)

            X_sample = np.array([new_input])
            predict_new_sample(id3_model, nb_model, classes, feature_names, X_sample)

        else:
            print("❌ Lựa chọn không hợp lệ!")

    # 8. Kết thúc
    print(f"\n{'=' * 70}")
    print("✅ HOÀN THÀNH!")
    print(f"{'=' * 70}")
    print("\n📊 Tóm tắt kết quả:")
    print(f"   🌳 ID3 Decision Tree: {acc_id3 * 100:.2f}%")
    print(f"   🧮 Naive Bayes:       {acc_nb * 100:.2f}%")
    print(f"\n💾 File đã tạo:")
    print(f"   - model_comparison.png")
    print("\n🎉 Cảm ơn bạn đã sử dụng chương trình!")
    print("=" * 70)


if __name__ == "__main__":
    main()