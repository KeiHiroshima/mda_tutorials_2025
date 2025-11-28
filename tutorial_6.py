"""
第6回演習：最適化手法の比較
目的：異なる最適化手法の特性を理解する
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 日本語フォント設定（環境に応じて調整）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ========================================
# 評価関数の定義
# ========================================

def simple_quadratic(x):
    """
    簡単な2次関数（1次元）
    f(x) = (x-3)^2 + 1
    最小値: f(3) = 1
    """
    return (x - 3)**2 + 1


def simple_quadratic_gradient(x):
    """
    simple_quadraticの勾配
    f'(x) = 2(x-3)
    """
    return 2 * (x - 3)


def rosenbrock(x, y):
    """
    Rosenbrock関数（2次元）
    f(x,y) = (1-x)^2 + 100(y-x^2)^2
    最小値: f(1, 1) = 0
    """
    return (1 - x)**2 + 100 * (y - x**2)**2


def rosenbrock_gradient(x, y):
    """
    Rosenbrock関数の勾配
    ∂f/∂x = -2(1-x) - 400x(y-x^2)
    ∂f/∂y = 200(y-x^2)
    """
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])


# ========================================
# 可視化関数
# ========================================

def plot_1d_function_and_points(func, x_min, x_max, x_history, title):
    """
    1次元関数と探索点をプロット
    """
    x = np.linspace(x_min, x_max, 300)
    y = func(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='f(x)')

    if len(x_history) > 0:
        y_history = [func(x_val) for x_val in x_history]
        plt.scatter(x_history, y_history, c='red', s=100,
                   alpha=0.6, label='Search points', zorder=5)

        # 最良点を強調
        best_idx = np.argmin(y_history)
        plt.scatter(x_history[best_idx], y_history[best_idx],
                   c='green', s=200, marker='*',
                   label='Best point', zorder=10)

    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_2d_contour_and_points(func, x_range, y_range, points_history, title):
    """
    2次元関数の等高線と探索軌跡をプロット
    """
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    plt.figure(figsize=(10, 8))

    # 等高線プロット（対数スケール）
    levels = np.logspace(-1, 3, 20)
    contour = plt.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, label='f(x, y)')

    # 探索軌跡をプロット
    if len(points_history) > 0:
        points = np.array(points_history)
        plt.plot(points[:, 0], points[:, 1], 'ro-',
                markersize=6, linewidth=1.5, alpha=0.7, label='Trajectory')
        plt.scatter(points[0, 0], points[0, 1], c='green', s=150,
                   marker='s', label='Start', zorder=10)
        plt.scatter(points[-1, 0], points[-1, 1], c='red', s=150,
                   marker='*', label='End', zorder=10)

    # 真の最小値をプロット
    plt.scatter(1, 1, c='blue', s=200, marker='x',
               linewidth=3, label='True minimum', zorder=10)

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ========================================
# ベースライン1：ランダムサーチ
# ========================================

def random_search_1d(func, x_min, x_max, n_trials, random_seed=42):
    """
    1次元ランダムサーチ

    Parameters:
    -----------
    func : callable
        目的関数
    x_min : float
        探索範囲の最小値
    x_max : float
        探索範囲の最大値
    n_trials : int
        試行回数
    random_seed : int
        乱数シード

    Returns:
    --------
    best_x : float
        最良の x 値
    best_value : float
        最良の関数値
    x_history : list
        探索した全ての x 値
    """
    np.random.seed(random_seed)

    # ランダムに点をサンプリング
    x_candidates = np.random.uniform(x_min, x_max, size=n_trials)

    # 最小値を持つ点を見つける
    best_x = None
    best_value = float('inf')

    for x in x_candidates:
        value = func(x)
        if value < best_value:
            best_x = x
            best_value = value

    return best_x, best_value, list(x_candidates)


def random_search_2d(func, x_range, y_range, n_trials, random_seed=42):
    """
    2次元ランダムサーチ

    Parameters:
    -----------
    func : callable
        目的関数 f(x, y)
    x_range : tuple
        x の探索範囲 (min, max)
    y_range : tuple
        y の探索範囲 (min, max)
    n_trials : int
        試行回数
    random_seed : int
        乱数シード

    Returns:
    --------
    best_point : array
        最良の点 [x, y]
    best_value : float
        最良の関数値
    points_history : list
        探索した全ての点
    """
    np.random.seed(random_seed)

    # ランダムに点をサンプリング
    x_candidates = np.random.uniform(x_range[0], x_range[1], size=n_trials)
    y_candidates = np.random.uniform(y_range[0], y_range[1], size=n_trials)

    # 最小値を持つ点を見つける
    best_point = None
    best_value = float('inf')
    points_history = []

    for x, y in zip(x_candidates, y_candidates):
        points_history.append([x, y])
        value = func(x, y)
        if value < best_value:
            best_point = np.array([x, y])
            best_value = value

    return best_point, best_value, points_history


# ========================================
# ベースライン2：グリッドサーチ
# ========================================

def grid_search_1d(func, x_min, x_max, n_points):
    """
    1次元グリッドサーチ

    Parameters:
    -----------
    func : callable
        目的関数
    x_min : float
        探索範囲の最小値
    x_max : float
        探索範囲の最大値
    n_points : int
        探索点の数

    Returns:
    --------
    best_x : float
        最良の x 値
    best_value : float
        最良の関数値
    x_history : list
        探索した全ての x 値
    """
    # 等間隔の点を生成
    x_candidates = np.linspace(x_min, x_max, n_points)

    # 最小値を持つ点を見つける
    best_x = None
    best_value = float('inf')

    for x in x_candidates:
        value = func(x)
        if value < best_value:
            best_x = x
            best_value = value

    return best_x, best_value, list(x_candidates)


def grid_search_2d(func, x_range, y_range, n_points_per_dim):
    """
    2次元グリッドサーチ

    Parameters:
    -----------
    func : callable
        目的関数 f(x, y)
    x_range : tuple
        x の探索範囲 (min, max)
    y_range : tuple
        y の探索範囲 (min, max)
    n_points_per_dim : int
        各次元あたりの探索点数

    Returns:
    --------
    best_point : array
        最良の点 [x, y]
    best_value : float
        最良の関数値
    points_history : list
        探索した全ての点
    """
    # 等間隔の点を生成
    x_candidates = np.linspace(x_range[0], x_range[1], n_points_per_dim)
    y_candidates = np.linspace(y_range[0], y_range[1], n_points_per_dim)

    # 最小値を持つ点を見つける
    best_point = None
    best_value = float('inf')
    points_history = []

    for x in x_candidates:
        for y in y_candidates:
            points_history.append([x, y])
            value = func(x, y)
            if value < best_value:
                best_point = np.array([x, y])
                best_value = value

    return best_point, best_value, points_history


# ========================================
# Black-box最適化：Nelder-Mead法
# ========================================

def nelder_mead_1d(func, x_init, max_iter=50):
    """
    1次元Nelder-Mead法
    （scipy.optimize.minimizeのラッパー）

    Parameters:
    -----------
    func : callable
        目的関数
    x_init : float
        初期値
    max_iter : int
        最大反復回数

    Returns:
    --------
    best_x : float
        最良の x 値
    best_value : float
        最良の関数値
    x_history : list
        探索履歴
    """
    from scipy.optimize import minimize

    # 履歴を記録するためのリスト
    x_history = []

    def callback(xk):
        x_history.append(float(xk))

    # 最適化実行
    result = minimize(func, x_init, method='Nelder-Mead',
                     callback=callback,
                     options={'maxiter': max_iter})

    return result.x[0], result.fun, x_history


def nelder_mead_2d(func_2d, x_init, y_init, max_iter=100):
    """
    2次元Nelder-Mead法

    Parameters:
    -----------
    func_2d : callable
        目的関数 f(x, y)
    x_init : float
        x の初期値
    y_init : float
        y の初期値
    max_iter : int
        最大反復回数

    Returns:
    --------
    best_point : array
        最良の点 [x, y]
    best_value : float
        最良の関数値
    points_history : list
        探索履歴
    """
    from scipy.optimize import minimize

    # 履歴を記録するためのリスト
    points_history = []

    def func_wrapper(point):
        return func_2d(point[0], point[1])

    def callback(xk):
        points_history.append(xk.copy())

    # 最適化実行
    x0 = np.array([x_init, y_init])
    result = minimize(func_wrapper, x0, method='Nelder-Mead',
                     callback=callback,
                     options={'maxiter': max_iter})

    return result.x, result.fun, points_history


# ========================================
# 勾配降下法
# ========================================

def gradient_descent_1d(func, grad_func, x_init, learning_rate,
                        n_iterations):
    """
    1次元勾配降下法

    Parameters:
    -----------
    func : callable
        目的関数
    grad_func : callable
        勾配関数
    x_init : float
        初期値
    learning_rate : float
        学習率
    n_iterations : int
        反復回数

    Returns:
    --------
    best_x : float
        最終的な x 値
    best_value : float
        最終的な関数値
    x_history : list
        探索履歴
    """
    x = x_init
    x_history = [x]

    for i in range(n_iterations):
        # 勾配を計算
        grad = grad_func(x)

        # パラメータを更新
        x = x - learning_rate * grad
        x_history.append(x)

    return x, func(x), x_history


def gradient_descent_2d(func, grad_func, x_init, y_init,
                        learning_rate, n_iterations):
    """
    2次元勾配降下法

    Parameters:
    -----------
    func : callable
        目的関数 f(x, y)
    grad_func : callable
        勾配関数（[∂f/∂x, ∂f/∂y]を返す）
    x_init : float
        x の初期値
    y_init : float
        y の初期値
    learning_rate : float
        学習率
    n_iterations : int
        反復回数

    Returns:
    --------
    best_point : array
        最終的な点 [x, y]
    best_value : float
        最終的な関数値
    points_history : list
        探索履歴
    """
    point = np.array([x_init, y_init])
    points_history = [point.copy()]

    for i in range(n_iterations):
        # 勾配を計算
        grad = grad_func(point[0], point[1])

        # パラメータを更新
        point = point - learning_rate * grad
        points_history.append(point.copy())

    return point, func(point[0], point[1]), points_history


# ========================================
# 演習1：1次元関数での比較
# ========================================

print("=" * 70)
print("演習1：1次元関数 f(x) = (x-3)^2 + 1 での最適化")
print("=" * 70)
print(f"真の最小値: x = 3.0, f(x) = 1.0")
print()

# 探索範囲の設定
x_min, x_max = -2, 8
n_trials = 20

# ランダムサーチ
print("-" * 70)
print("1. ランダムサーチ")
print("-" * 70)
best_x, best_value, x_history = random_search_1d(
    simple_quadratic, x_min, x_max, n_trials
)
print(f"探索点数: {len(x_history)}")
print(f"最良の x: {best_x:.4f}")
print(f"最良の f(x): {best_value:.4f}")
print(f"真の最小値からの誤差: {abs(best_x - 3.0):.4f}")
print()

plot_1d_function_and_points(
    simple_quadratic, x_min, x_max, x_history,
    'Random Search (1D)'
)

# グリッドサーチ
print("-" * 70)
print("2. グリッドサーチ")
print("-" * 70)
best_x, best_value, x_history = grid_search_1d(
    simple_quadratic, x_min, x_max, n_trials
)
print(f"探索点数: {len(x_history)}")
print(f"最良の x: {best_x:.4f}")
print(f"最良の f(x): {best_value:.4f}")
print(f"真の最小値からの誤差: {abs(best_x - 3.0):.4f}")
print()

plot_1d_function_and_points(
    simple_quadratic, x_min, x_max, x_history,
    'Grid Search (1D)'
)

# Nelder-Mead法
print("-" * 70)
print("3. Black-box最適化（Nelder-Mead法）")
print("-" * 70)
best_x, best_value, x_history = nelder_mead_1d(
    simple_quadratic, x_init=0.0, max_iter=20
)
print(f"反復回数: {len(x_history)}")
print(f"最良の x: {best_x:.4f}")
print(f"最良の f(x): {best_value:.4f}")
print(f"真の最小値からの誤差: {abs(best_x - 3.0):.4f}")
print()

plot_1d_function_and_points(
    simple_quadratic, x_min, x_max, x_history,
    'Nelder-Mead Method (1D)'
)

# 勾配降下法
print("-" * 70)
print("4. 勾配降下法")
print("-" * 70)
best_x, best_value, x_history = gradient_descent_1d(
    simple_quadratic, simple_quadratic_gradient,
    x_init=0.0, learning_rate=0.1, n_iterations=20
)
print(f"反復回数: {len(x_history) - 1}")
print(f"最良の x: {best_x:.4f}")
print(f"最良の f(x): {best_value:.4f}")
print(f"真の最小値からの誤差: {abs(best_x - 3.0):.4f}")
print()

plot_1d_function_and_points(
    simple_quadratic, x_min, x_max, x_history,
    'Gradient Descent (1D, learning_rate=0.1)'
)


# ========================================
# 演習2：2次元Rosenbrock関数での比較
# ========================================

print("\n" + "=" * 70)
print("演習2：2次元Rosenbrock関数での最適化")
print("=" * 70)
print(f"真の最小値: x = 1.0, y = 1.0, f(x,y) = 0.0")
print()

# 探索範囲の設定
x_range = (-2, 2)
y_range = (-1, 3)
n_trials_2d = 50
n_grid_per_dim = 10  # グリッドサーチ：10x10 = 100点

# ランダムサーチ
print("-" * 70)
print("1. ランダムサーチ（2次元）")
print("-" * 70)
best_point, best_value, points_history = random_search_2d(
    rosenbrock, x_range, y_range, n_trials_2d
)
print(f"探索点数: {len(points_history)}")
print(f"最良の点: x = {best_point[0]:.4f}, y = {best_point[1]:.4f}")
print(f"最良の f(x,y): {best_value:.4f}")
print(f"真の最小値からの距離: {np.linalg.norm(best_point - np.array([1, 1])):.4f}")
print()

plot_2d_contour_and_points(
    rosenbrock, x_range, y_range, points_history,
    'Random Search (Rosenbrock)'
)

# グリッドサーチ
print("-" * 70)
print("2. グリッドサーチ（2次元）")
print("-" * 70)
best_point, best_value, points_history = grid_search_2d(
    rosenbrock, x_range, y_range, n_grid_per_dim
)
print(f"探索点数: {len(points_history)}")
print(f"最良の点: x = {best_point[0]:.4f}, y = {best_point[1]:.4f}")
print(f"最良の f(x,y): {best_value:.4f}")
print(f"真の最小値からの距離: {np.linalg.norm(best_point - np.array([1, 1])):.4f}")
print()

plot_2d_contour_and_points(
    rosenbrock, x_range, y_range, points_history,
    'Grid Search (Rosenbrock)'
)

# Nelder-Mead法
print("-" * 70)
print("3. Black-box最適化（Nelder-Mead法、2次元）")
print("-" * 70)
best_point, best_value, points_history = nelder_mead_2d(
    rosenbrock, x_init=-1.0, y_init=0.0, max_iter=100
)
print(f"反復回数: {len(points_history)}")
print(f"最良の点: x = {best_point[0]:.4f}, y = {best_point[1]:.4f}")
print(f"最良の f(x,y): {best_value:.4f}")
print(f"真の最小値からの距離: {np.linalg.norm(best_point - np.array([1, 1])):.4f}")
print()

plot_2d_contour_and_points(
    rosenbrock, x_range, y_range, points_history,
    'Nelder-Mead Method (Rosenbrock)'
)

# 勾配降下法
print("-" * 70)
print("4. 勾配降下法（2次元）")
print("-" * 70)
best_point, best_value, points_history = gradient_descent_2d(
    rosenbrock, rosenbrock_gradient,
    x_init=-1.0, y_init=0.0, learning_rate=0.001, n_iterations=1000
)
print(f"反復回数: {len(points_history) - 1}")
print(f"最良の点: x = {best_point[0]:.4f}, y = {best_point[1]:.4f}")
print(f"最良の f(x,y): {best_value:.4f}")
print(f"真の最小値からの距離: {np.linalg.norm(best_point - np.array([1, 1])):.4f}")
print()

plot_2d_contour_and_points(
    rosenbrock, x_range, y_range, points_history,
    'Gradient Descent (Rosenbrock, learning_rate=0.001)'
)


# ========================================
# 演習3：性能比較
# ========================================

print("\n" + "=" * 70)
print("演習3：性能比較（100回試行の平均）")
print("=" * 70)

n_experiments = 100

# 1次元関数での比較
print("\n--- 1次元関数 f(x) = (x-3)^2 + 1 ---")

random_errors_1d = []
grid_errors_1d = []
nm_errors_1d = []
gd_errors_1d = []

for i in range(n_experiments):
    # ランダムサーチ
    best_x, _, _ = random_search_1d(
        simple_quadratic, -2, 8, 20, random_seed=i
    )
    random_errors_1d.append(abs(best_x - 3.0))

    # グリッドサーチ（固定のため1回のみ計算）
    if i == 0:
        best_x, _, _ = grid_search_1d(simple_quadratic, -2, 8, 20)
        grid_error_1d = abs(best_x - 3.0)

    # Nelder-Mead法
    best_x, _, _ = nelder_mead_1d(
        simple_quadratic, x_init=np.random.uniform(-2, 8), max_iter=20
    )
    nm_errors_1d.append(abs(best_x - 3.0))

    # 勾配降下法
    best_x, _, _ = gradient_descent_1d(
        simple_quadratic, simple_quadratic_gradient,
        x_init=np.random.uniform(-2, 8), learning_rate=0.1, n_iterations=20
    )
    gd_errors_1d.append(abs(best_x - 3.0))

print(f"ランダムサーチ    平均誤差: {np.mean(random_errors_1d):.6f} ± {np.std(random_errors_1d):.6f}")
print(f"グリッドサーチ    平均誤差: {grid_error_1d:.6f}")
print(f"Nelder-Mead法     平均誤差: {np.mean(nm_errors_1d):.6f} ± {np.std(nm_errors_1d):.6f}")
print(f"勾配降下法        平均誤差: {np.mean(gd_errors_1d):.6f} ± {np.std(gd_errors_1d):.6f}")

# 2次元Rosenbrock関数での比較
print("\n--- 2次元Rosenbrock関数 ---")

random_errors_2d = []
nm_errors_2d = []
gd_errors_2d = []

for i in range(n_experiments):
    # ランダムサーチ
    best_point, _, _ = random_search_2d(
        rosenbrock, (-2, 2), (-1, 3), 50, random_seed=i
    )
    random_errors_2d.append(np.linalg.norm(best_point - np.array([1, 1])))

    # グリッドサーチ（固定のため1回のみ計算）
    if i == 0:
        best_point, _, _ = grid_search_2d(
            rosenbrock, (-2, 2), (-1, 3), 10
        )
        grid_error_2d = np.linalg.norm(best_point - np.array([1, 1]))

    # Nelder-Mead法
    x_init = np.random.uniform(-2, 2)
    y_init = np.random.uniform(-1, 3)
    best_point, _, _ = nelder_mead_2d(
        rosenbrock, x_init, y_init, max_iter=100
    )
    nm_errors_2d.append(np.linalg.norm(best_point - np.array([1, 1])))

    # 勾配降下法
    x_init = np.random.uniform(-2, 2)
    y_init = np.random.uniform(-1, 3)
    best_point, _, _ = gradient_descent_2d(
        rosenbrock, rosenbrock_gradient,
        x_init, y_init, learning_rate=0.001, n_iterations=1000
    )
    gd_errors_2d.append(np.linalg.norm(best_point - np.array([1, 1])))

print(f"ランダムサーチ    平均誤差: {np.mean(random_errors_2d):.6f} ± {np.std(random_errors_2d):.6f}")
print(f"グリッドサーチ    平均誤差: {grid_error_2d:.6f}")
print(f"Nelder-Mead法     平均誤差: {np.mean(nm_errors_2d):.6f} ± {np.std(nm_errors_2d):.6f}")
print(f"勾配降下法        平均誤差: {np.mean(gd_errors_2d):.6f} ± {np.std(gd_errors_2d):.6f}")


# ========================================
# 考察課題
# ========================================

print("\n" + "=" * 70)
print("考察課題")
print("=" * 70)
print("""
1. ランダムサーチとグリッドサーチの違いを説明せよ。
   それぞれの利点と欠点は何か。

2. Nelder-Mead法（Black-box最適化）は、ランダムサーチや
   グリッドサーチと比較してどのような特徴があるか。

3. 勾配降下法が他の手法と比較して優れている点は何か。
   また、勾配情報を利用することの利点を説明せよ。

4. 1次元問題と2次元Rosenbrock関数で、各手法の性能が
   どのように変化したか。次元が増えると最適化が
   困難になる理由を考察せよ。

5. 勾配降下法の学習率を変更すると、収束速度や
   安定性がどのように変化するか実験せよ。
""")
