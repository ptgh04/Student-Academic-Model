import pandas as pd
import numpy as np
from collections import Counter
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

class ID3DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = None

    def entropy(self, y):
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
        parent_entropy = self.entropy(y)
        n = len(y)
        if threshold is not None:
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
        X_col = X[:, feature_idx]
        unique_values = np.unique(X_col)
        if len(unique_values) <= 10:
            ig = self.information_gain(X_col, y)
            return ig, None
        else:
            sorted_values = np.sort(unique_values)
            candidate_thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
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
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                n_classes == 1):
            leaf_value = Counter(y).most_common(1)[0][0]
            return {
                'leaf': True,
                'value': leaf_value,
                'samples': n_samples
            }
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        for feature_idx in range(n_features):
            gain, threshold = self.find_best_split(X, y, feature_idx)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
        if best_gain == 0.0 or best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {
                'leaf': True,
                'value': leaf_value,
                'samples': n_samples
            }
        X_col = X[:, best_feature]
        if best_threshold is not None:
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
        print("Bat dau huan luyen ID3 Decision Tree...")
        self.feature_names = feature_names
        self.tree = self.build_tree(X, y)
        print("Huan luyen ID3 hoan tat.")
        return self

    def predict_sample(self, x, tree):
        if tree['leaf']:
            return tree['value']
        feature_val = x[tree['feature']]
        if tree['threshold'] is not None:
            if feature_val <= tree['threshold']:
                return self.predict_sample(x, tree['left'])
            else:
                return self.predict_sample(x, tree['right'])
        else:
            if feature_val in tree['branches']:
                return self.predict_sample(x, tree['branches'][feature_val])
            else:
                return tree.get('default_class', 0)

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self.predict_sample(x, self.tree)
            predictions.append(pred)
        return np.array(predictions)

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.means = {}
        self.variances = {}
        self.epsilon = 1e-9

    def fit(self, X, y):
        print("Bat dau huan luyen Naive Bayes...")
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / n_samples
            self.means[c] = np.mean(X_c, axis=0)
            self.variances[c] = np.var(X_c, axis=0) + self.epsilon
        print("Huan luyen Naive Bayes hoan tat.")
        return self

    def gaussian_probability(self, x, mean, var):
        coefficient = 1.0 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coefficient * exponent

    def predict_sample(self, x):
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
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        predictions = []
        for x in X:
            pred = self.predict_sample(x)
            predictions.append(pred)
        return np.array(predictions)

    def predict_proba(self, X):
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
            max_log_prob = max(posteriors.values())
            exp_probs = {c: np.exp(log_prob - max_log_prob)
                         for c, log_prob in posteriors.items()}
            total = sum(exp_probs.values())
            probs = np.array([exp_probs[c] / total for c in self.classes])
            probabilities.append(probs)
        return np.array(probabilities)

def load_and_preprocess_data(file_path):
    print(f"\n{'=' * 70}")
    print("XU LY DU LIEU")
    print(f"{'=' * 70}\n")
    print(f"Doc file: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Da doc {len(df)} mau voi {len(df.columns)} cot")
    print(f"\nCac cot trong dataset:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        print(f"   {i:2d}. {col:30s} - Type: {dtype}, Unique: {n_unique}")
    print(f"\nXu ly du lieu thieu...")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"   Tim thay {missing_counts.sum()} gia tri thieu")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print(f"   Khong co du lieu thieu")
    possible_targets = ['Target', 'target', 'label', 'class', 'output']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
    print(f"\nCot target: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = X.columns.tolist()
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded: {col}")
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    print(f"\nPhan bo target:")
    for i, class_name in enumerate(le_target.classes_):
        count = sum(y_encoded == i)
        print(f"   {class_name:20s}: {count:5d} ({count / len(y_encoded) * 100:.2f}%)")
    return X.values, y_encoded, le_target.classes_, feature_names

def calculate_accuracy(y_true, y_pred):
    correct = sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

def calculate_metrics(y_true, y_pred, classes):
    metrics = {}
    for i, class_name in enumerate(classes):
        tp = sum((y_true == i) & (y_pred == i))
        fp = sum((y_true != i) & (y_pred == i))
        fn = sum((y_true == i) & (y_pred != i))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
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
    print(f"\n{'=' * 70}")
    print(f"KET QUA MO HINH: {model_name}")
    print(f"{'=' * 70}")
    accuracy = calculate_accuracy(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    metrics = calculate_metrics(y_true, y_pred, classes)
    print(f"\nChi tiet tung class:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)
    for class_name, metric in metrics.items():
        print(f"{class_name:<15} "
              f"{metric['precision']:>10.4f} "
              f"{metric['recall']:>10.4f} "
              f"{metric['f1-score']:>10.4f} "
              f"{metric['support']:>10d}")
    macro_precision = np.mean([m['precision'] for m in metrics.values()])
    macro_recall = np.mean([m['recall'] for m in metrics.values()])
    macro_f1 = np.mean([m['f1-score'] for m in metrics.values()])
    print("-" * 70)
    print(f"{'Macro Avg':<15} "
          f"{macro_precision:>10.4f} "
          f"{macro_recall:>10.4f} "
          f"{macro_f1:>10.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    return accuracy, cm

def plot_comparison(accuracies, cms, classes):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = list(accuracies.keys())
    accs = list(accuracies.values())
    colors = ['#3498db', '#e74c3c']
    bars = axes[0].bar(models, accs, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Comparison of Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    for i, (bar, v) in enumerate(zip(bars, accs)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{v:.4f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=11)
    for idx, (model_name, cm) in enumerate(cms.items(), start=1):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes,
                    ax=axes[idx], cbar_kws={'label': 'Count'},
                    linewidths=0.5, linecolor='gray')
        axes[idx].set_title(f'Confusion Matrix - {model_name}',
                            fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nDa luu bieu do: model_comparison.png")
    plt.show()

def predict_new_sample(id3_model, nb_model, classes, feature_names, X_sample):
    print(f"\n{'=' * 70}")
    print("KET QUA DU DOAN")
    print(f"{'=' * 70}")
    pred_id3 = id3_model.predict(X_sample)[0]
    print(f"\nID3 Decision Tree: {classes[pred_id3]}")
    pred_nb = nb_model.predict(X_sample)[0]
    pred_proba = nb_model.predict_proba(X_sample)[0]
    print(f"Naive Bayes: {classes[pred_nb]}")
    print(f"\nXac suat du doan (Naive Bayes):")
    for i, class_name in enumerate(classes):
        print(f"   {class_name:20s} {pred_proba[i] * 100:.2f}%")
    if pred_id3 == pred_nb:
        print(f"\nHai mo hinh dong y: {classes[pred_id3]}")
    else:
        print(f"\nHai mo hinh khac nhau:")
        print(f"   - ID3: {classes[pred_id3]}")
        print(f"   - Naive Bayes: {classes[pred_nb]}")
    return pred_id3, pred_nb

def main():
    print("=" * 70)
    print("DU DOAN DROPOUT & ACADEMIC SUCCESS")
    print("=" * 70)
    print("Thuat toan: ID3 Decision Tree + Naive Bayes")
    print("=" * 70)
    file_path = input("\nNhap duong dan file CSV: ").strip()
    if not file_path:
        file_path = 'dataset.csv'
    try:
        X, y, classes, feature_names = load_and_preprocess_data(file_path)
    except Exception as e:
        print(f"\nLoi doc file: {e}")
        return
    print(f"\n{'=' * 70}")
    print("CHIA DU LIEU TRAIN/TEST")
    print(f"{'=' * 70}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nKet qua:")
    print(f"   - Train: {len(X_train)} mau")
    print(f"   - Test:  {len(X_test)} mau")
    print(f"\n{'=' * 70}")
    print("HUAN LUYEN ID3 DECISION TREE")
    print(f"{'=' * 70}")
    id3_model = ID3DecisionTree(max_depth=10, min_samples_split=5)
    id3_model.fit(X_train, y_train, feature_names)
    print("\nDu doan tren tap test...")
    y_pred_id3 = id3_model.predict(X_test)
    acc_id3, cm_id3 = evaluate_model(y_test, y_pred_id3, "ID3 Decision Tree", classes)
    print(f"\n{'=' * 70}")
    print("HUAN LUYEN NAIVE BAYES")
    print(f"{'=' * 70}")
    nb_model = NaiveBayes()
    nb_model.fit(X_train, y_train)
    print("\nDu doan tren tap test...")
    y_pred_nb = nb_model.predict(X_test)
    acc_nb, cm_nb = evaluate_model(y_test, y_pred_nb, "Naive Bayes", classes)
    print(f"\n{'=' * 70}")
    print("TONG KET SO SANH")
    print(f"{'=' * 70}")
    print(f"\nID3 Decision Tree: {acc_id3:.4f} ({acc_id3 * 100:.2f}%)")
    print(f"Naive Bayes:       {acc_nb:.4f} ({acc_nb * 100:.2f}%)")
    diff = abs(acc_id3 - acc_nb)
    if acc_id3 > acc_nb:
        print(f"\nID3 Decision Tree tot hon {diff * 100:.2f}%")
    elif acc_nb > acc_id3:
        print(f"\nNaive Bayes tot hon {diff * 100:.2f}%")
    else:
        print(f"\nHai mo hinh co do chinh xac tuong duong")
    print(f"\n{'=' * 70}")
    print("TAO BIEU DO SO SANH")
    print(f"{'=' * 70}")
    accuracies = {'ID3 Decision Tree': acc_id3, 'Naive Bayes': acc_nb}
    cms = {'ID3 Decision Tree': cm_id3, 'Naive Bayes': cm_nb}
    plot_comparison(accuracies, cms, classes)
    print(f"\n{'=' * 70}")
    print("DU DOAN DU LIEU MOI")
    print(f"{'=' * 70}")
    while True:
        print("\nChon cach du doan:")
        print("   1. Nhap thu cong")
        print("   2. Dung mau random tu dataset")
        print("   3. In cau truc cay ID3")
        print("   4. Thoat")
        choice = input("\nLua chon (1/2/3/4): ").strip()
        if choice == '4':
            break
        elif choice == '3':
            print(f"\n{'=' * 70}")
            print("CAU TRUC CAY QUYET DINH ID3")
            print(f"{'=' * 70}\n")
            depth_limit = input("Do sau toi da (Enter = 3): ").strip()
            depth_limit = int(depth_limit) if depth_limit else 3
            def print_tree_limited(tree, depth=0, prefix="Root", max_depth=3):
                if depth >= max_depth:
                    return
                indent = "  " * depth
                if tree['leaf']:
                    print(f"{indent}{prefix}: Leaf -> Class {classes[tree['value']]} "
                          f"(samples: {tree['samples']})")
                else:
                    feature_name = (feature_names[tree['feature']]
                                    if feature_names else f"Feature {tree['feature']}")
                    if tree['threshold'] is not None:
                        print(f"{indent}{prefix}: {feature_name} <= {tree['threshold']:.2f} "
                              f"(IG: {tree['gain']:.4f}, samples: {tree['samples']})")
                        print_tree_limited(tree['left'], depth + 1, "Left ", max_depth)
                        print_tree_limited(tree['right'], depth + 1, "Right", max_depth)
                    else:
                        print(f"{indent}{prefix}: {feature_name} "
                              f"(IG: {tree['gain']:.4f}, samples: {tree['samples']})")
                        branches = list(tree['branches'].items())
                        for i, (value, subtree) in enumerate(branches):
                            print_tree_limited(subtree, depth + 1, f"Val={value}", max_depth)
            print_tree_limited(id3_model.tree, max_depth=depth_limit)
            continue
        elif choice == '2':
            random_idx = np.random.randint(0, len(X))
            X_sample = X[random_idx:random_idx + 1]
            print(f"\nMau ngau nhien (index: {random_idx}):")
            print(f"\nGia tri cac dac trung:")
            for i, (fname, val) in enumerate(zip(feature_names, X_sample[0])):
                if i < 10:
                    print(f"   {i + 1:2d}. {fname:30s} = {val:.2f}")
            if len(feature_names) > 10:
                print(f"   ... va {len(feature_names) - 10} features khac")
            predict_new_sample(id3_model, nb_model, classes, feature_names, X_sample)
        elif choice == '1':
            print(f"\nNhap du lieu cho {len(feature_names)} dac trung")
            print("Nhan Enter de dung gia tri trung binh")
            print("Nhap 'skip' de bo qua cac features con lai\n")
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
                    print(f"     Dung gia tri trung binh cho cac features con lai")
                    new_input.append(mean_val)
                    skip_rest = True
                elif user_input == "":
                    new_input.append(mean_val)
                else:
                    try:
                        new_input.append(float(user_input))
                    except:
                        print(f"     Loi, dung mean: {mean_val:.2f}")
                        new_input.append(mean_val)
            X_sample = np.array([new_input])
            predict_new_sample(id3_model, nb_model, classes, feature_names, X_sample)
        else:
            print("Lua chon khong hop le!")
    print(f"\n{'=' * 70}")
    print("HOAN THANH")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()