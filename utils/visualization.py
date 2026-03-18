import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.manifold import TSNE
import json
from sklearn.decomposition import PCA
import torch
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.offline import plot
import umap
import pacmap

def fuse_labels(labels):
    """
    将原始五分类标签融合为三分类标签：
        N      -> 0
        L1     -> 1
        L2/L3/L2L3 -> 2

    原始标签编号默认约定：
        0 -> N
        1 -> L2
        2 -> L3
        3 -> L2L3
        4 -> L1
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    labels = np.asarray(labels)
    fused = np.full_like(labels, fill_value=-1)

    fused[labels == 0] = 0          # N
    fused[labels == 1] = 1          # L1
    fused[(labels == 2) | (labels == 3) | (labels == 4)] = 2   # L2/L3/L2L3

    if np.any(fused == -1):
        invalid = np.unique(labels[fused == -1])
        raise ValueError(f"发现未定义标签: {invalid}")

    return fused

def plot_tsne(
    X,
    y,
    unique_classes=("N", "L2", "L3", "L2L3", "L1"),
    save_path=None,
    font_path="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    figsize=(6.5, 6),
    dpi=400,
    random_state=36,
    perplexity=30,
    n_iter=1000,
    point_size=20,
    alpha=1.0,
):
    """
    绘制 t-SNE 可视化图

    参数
    ----
    X : np.ndarray, shape [N, D]
        输入特征
    y : np.ndarray or list, shape [N]
        标签，可为字符串标签或整数标签
    unique_classes : list/tuple
        类别顺序
    save_path : str
        图片保存路径
    font_path : str
        Linux 下中文字体路径
    figsize : tuple
        图像尺寸
    dpi : int
        保存分辨率
    random_state : int
        t-SNE 随机种子
    perplexity : int/float
        t-SNE perplexity
    n_iter : int
        t-SNE 迭代次数
    point_size : int/float
        散点大小
    alpha : float
        散点透明度
    """

    # ========= 1. 检查字体 =========
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件不存在: {font_path}")

    font_prop = font_manager.FontProperties(fname=font_path)

    # 注册字体，避免某些 Linux 环境下中文失效
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

    # ========= 2. 数据检查 =========
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X 应为二维数组 [N, D]，当前形状为 {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X 和 y 长度不一致: len(X)={len(X)}, len(y)={len(y)}")

    # ========= 3. 如果标签是字符串，映射为统一顺序 =========
    # 支持 y 中直接存 "N"、"L2" 这种字符串
    if y.dtype.kind in {"U", "S", "O"}:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        y_idx = np.array([class_to_idx[label] for label in y])
    else:
        # 若 y 已经是整数标签，则默认按照 unique_classes 的索引顺序使用
        y_idx = y.astype(int)

    # ========= 4. t-SNE 降维 =========
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=n_iter,
        random_state=random_state,
    )
    X_tsne = tsne.fit_transform(X)

    # ========= 5. 颜色设置 =========
    # 5 类足够清晰的一组颜色
    colors = [
        "#1f77b4",  # 蓝
        "#ff7f0e",  # 橙
        "#2ca02c",  # 绿
        "#d62728",  # 红
        "#9467bd",  # 紫
    ]

    # ========= 6. 开始绘图 =========
    fig, ax = plt.subplots(figsize=figsize)

    for i, cls_name in enumerate(unique_classes):
        mask = (y_idx == i)
        if np.sum(mask) == 0:
            continue

        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=point_size,
            c=colors[i % len(colors)],
            label=f"故障编码：{cls_name}",
            alpha=alpha,
            linewidths=0,
            marker="o",
        )

    # ========= 7. 坐标轴与风格 =========
    ax.set_xlabel("t-SNE特征1", fontproperties=font_prop, fontsize=13)
    ax.set_ylabel("t-SNE特征2", fontproperties=font_prop, fontsize=13)

    # 坐标轴刻度字体
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(11)

    # 边框稍微清晰一点
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # ========= 8. 图例放到图外下方，避免遮挡 =========
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),   # 放在图外下方
        ncol=3,                        # 多列排列
        frameon=True,
        prop=font_prop,
        fontsize=10,
        handletextpad=0.4,
        columnspacing=1.6,
        borderpad=0.5,
    )

    # 给图例边框一点样式
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)

    # 让底部留出足够空间放图例
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    # ========= 9. 保存 =========
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return X_tsne

def plot_tsne_fused(
    features,
    labels,
    save_path=None,
    font_path="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    figsize=(6.5, 6.0),
    dpi=400,
    random_state=42,
    perplexity=20,
    n_iter=1500,
    point_size=35,
    alpha=1.0,
):
    """
    按照融合标签逻辑绘制 t-SNE 图
    不修改原 plot_tsne 的任何配置，只是在外部先做标签映射
    """
    fused_labels = fuse_labels(labels)

    fused_class_names = ("N", "L1", "L2/L3/L2L3")

    return plot_tsne(
        X=features,
        y=fused_labels,
        unique_classes=fused_class_names,
        save_path=save_path,
        font_path=font_path,
        figsize=figsize,
        dpi=dpi,
        random_state=random_state,
        perplexity=perplexity,
        n_iter=n_iter,
        point_size=point_size,
        alpha=alpha,
    )

def plot_tsne_3d(
    X,
    y,
    unique_classes=("N", "L1", "L2", "L3", "L2L3"),
    save_path=None,
    font_path="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    figsize=(6.8, 5.4),
    dpi=400,
    random_state=36,
    perplexity=30,
    n_iter=1000,
    point_size=26,
    alpha=0.95,
    elev=20,
    azim=35,
    show=False,
):
    # ========= 1. 检查字体 =========
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件不存在: {font_path}")

    font_prop = font_manager.FontProperties(fname=font_path)
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False

    # ========= 2. 数据检查 =========
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X 应为二维数组 [N, D]，当前形状为 {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X 和 y 长度不一致: len(X)={len(X)}, len(y)={len(y)}")
    if len(X) < 3:
        raise ValueError(f"样本数量过少，至少需要 3 个样本，当前为 {len(X)}")
    if np.isnan(X).any():
        raise ValueError("X 中存在 NaN，请先清理数据")
    if np.isinf(X).any():
        raise ValueError("X 中存在 Inf，请先清理数据")

    if perplexity >= len(X):
        perplexity = max(1, len(X) // 3)
        print(f"[Warning] perplexity 过大，已自动调整为 {perplexity}")

    # ========= 3. 标签映射 =========
    if y.dtype.kind in {"U", "S", "O"}:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        try:
            y_idx = np.array([class_to_idx[label] for label in y])
        except KeyError as e:
            raise ValueError(f"y 中存在未在 unique_classes 中定义的标签: {e}")
    else:
        y_idx = y.astype(int)

    # ========= 4. t-SNE =========
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=n_iter,
        random_state=random_state,
    )
    X_tsne = tsne.fit_transform(X)

    # ========= 5. 配色 =========
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # ========= 6. 创建图 =========
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # 关键1：直接手动放大 3D 坐标轴区域
    # [left, bottom, width, height]
    ax = fig.add_axes([0.02, 0.20, 0.96, 0.95], projection="3d")

    for i, cls_name in enumerate(unique_classes):
        mask = (y_idx == i)
        if np.sum(mask) == 0:
            continue

        ax.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            X_tsne[mask, 2],
            s=point_size,
            c=colors[i % len(colors)],
            label=f"故障编码：{cls_name}",
            alpha=alpha,
            linewidths=0.30,
            edgecolors="black",
            marker="o",
            depthshade=True,
        )

    # ========= 7. 坐标轴 =========
    ax.set_xlabel("t-SNE特征1", fontproperties=font_prop, fontsize=12, labelpad=2)
    ax.set_ylabel("t-SNE特征2", fontproperties=font_prop, fontsize=12, labelpad=2)

    # 关键2：显式控制 zlabel 位置
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("t-SNE特征3", fontproperties=font_prop, fontsize=12, rotation=90, labelpad=1)

    # 强制把 zlabel 往图内拉一点
    try:
        ax.zaxis.set_label_coords(-0.95, 0.5)
    except Exception:
        pass

    # 刻度字体
    for tick in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        tick.set_fontproperties(font_prop)
        tick.set_fontsize(10)

    # 视角
    ax.view_init(elev=elev, azim=azim)

    # 网格
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    # 关键3：压缩顶部空白，让三维盒子更“铺开”
    try:
        ax.set_box_aspect((1.05, 0.85, 1.05))
    except Exception:
        pass

    # ========= 8. 图例 =========
    legend = fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=3,
        frameon=True,
        prop=font_prop,
        fontsize=10,
        handletextpad=0.45,
        columnspacing=1.4,
        borderpad=0.35,
        labelspacing=0.55,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_alpha(1.0)

    # ========= 9. 保存 =========
    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.015
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return X_tsne

def plot_tsne_3d_html(
    X,
    y,
    unique_classes=("N", "L1", "L2", "L3", "L2L3"),
    save_path=None,
    random_state=36,
    perplexity=30,
    n_iter=1000,
    point_size=5,
    opacity=0.9,
    title="3维 t-SNE 可视化"
):
    """
    生成可交互旋转的 3D t-SNE HTML 文件

    参数
    ----
    X : np.ndarray, shape [N, D]
        输入特征
    y : np.ndarray or list, shape [N]
        标签，可为字符串或整数
    unique_classes : tuple/list
        类别顺序
    save_path : str
        输出 html 文件路径
    random_state : int
        t-SNE 随机种子
    perplexity : int/float
        t-SNE perplexity
    n_iter : int
        t-SNE 迭代轮数
    point_size : int/float
        散点大小
    opacity : float
        散点透明度
    title : str
        图标题
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X 应为二维数组 [N, D]，当前形状为 {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X 和 y 长度不一致: len(X)={len(X)}, len(y)={len(y)}")
    if len(X) < 3:
        raise ValueError("样本数过少，至少需要 3 个样本")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X 中存在 NaN 或 Inf，请先清理数据")

    if perplexity >= len(X):
        perplexity = max(1, len(X) // 3)
        print(f"[Warning] perplexity 过大，已自动调整为 {perplexity}")

    # 标签映射
    if y.dtype.kind in {"U", "S", "O"}:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        try:
            y_idx = np.array([class_to_idx[label] for label in y])
        except KeyError as e:
            raise ValueError(f"y 中存在未在 unique_classes 中定义的标签: {e}")
    else:
        y_idx = y.astype(int)

    # t-SNE
    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=n_iter,
        random_state=random_state,
    )
    X_tsne = tsne.fit_transform(X)

    # 颜色
    colors = [
        "#1f77b4",  # 蓝
        "#ff7f0e",  # 橙
        "#2ca02c",  # 绿
        "#d62728",  # 红
        "#9467bd",  # 紫
    ]

    traces = []
    for i, cls_name in enumerate(unique_classes):
        mask = (y_idx == i)
        if np.sum(mask) == 0:
            continue

        traces.append(
            go.Scatter3d(
                x=X_tsne[mask, 0],
                y=X_tsne[mask, 1],
                z=X_tsne[mask, 2],
                mode="markers",
                name=f"故障编码：{cls_name}",
                marker=dict(
                    size=point_size,
                    color=colors[i % len(colors)],
                    opacity=opacity,
                    line=dict(width=0.5, color="black"),
                ),
                text=[f"类别: {cls_name}"] * np.sum(mask),
                hovertemplate=(
                    "类别: %{text}<br>"
                    "t-SNE特征1: %{x:.3f}<br>"
                    "t-SNE特征2: %{y:.3f}<br>"
                    "t-SNE特征3: %{z:.3f}<extra></extra>"
                ),
            )
        )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="t-SNE特征1",
            yaxis_title="t-SNE特征2",
            zaxis_title="t-SNE特征3",
            bgcolor="white",
            xaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
            aspectmode="data",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=0, r=0, t=50, b=60),
        template="plotly_white",
    )

    save_path = os.fspath(save_path)

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plot(fig, filename=save_path, auto_open=False, include_plotlyjs=True)

    return X_tsne

def plot_umap_3d_html(
    X,
    y,
    unique_classes=("N", "L1", "L2", "L3", "L2L3"),
    save_path=None,
    random_state=36,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    point_size=5,
    opacity=0.9,
    title="3维 UMAP 可视化"
):
    """
    生成可交互旋转的 3D UMAP HTML 文件

    参数
    ----
    X : np.ndarray, shape [N, D]
        输入特征
    y : np.ndarray or list, shape [N]
        标签，可为字符串或整数
    unique_classes : tuple/list
        类别顺序
    save_path : str or Path
        输出 html 文件路径
    random_state : int
        随机种子
    n_neighbors : int
        UMAP 邻居数
    min_dist : float
        UMAP 最小距离
    metric : str
        距离度量
    point_size : int/float
        散点大小
    opacity : float
        散点透明度
    title : str
        图标题
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X 应为二维数组 [N, D]，当前形状为 {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X 和 y 长度不一致: len(X)={len(X)}, len(y)={len(y)}")
    if len(X) < 3:
        raise ValueError("样本数过少，至少需要 3 个样本")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X 中存在 NaN 或 Inf，请先清理数据")

    # 标签映射
    if y.dtype.kind in {"U", "S", "O"}:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        try:
            y_idx = np.array([class_to_idx[label] for label in y])
        except KeyError as e:
            raise ValueError(f"y 中存在未在 unique_classes 中定义的标签: {e}")
    else:
        y_idx = y.astype(int)

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    X_umap = reducer.fit_transform(X)

    colors = [
        "#1f77b4",  # 蓝
        "#ff7f0e",  # 橙
        "#2ca02c",  # 绿
        "#d62728",  # 红
        "#9467bd",  # 紫
    ]

    traces = []
    for i, cls_name in enumerate(unique_classes):
        mask = (y_idx == i)
        if np.sum(mask) == 0:
            continue

        traces.append(
            go.Scatter3d(
                x=X_umap[mask, 0],
                y=X_umap[mask, 1],
                z=X_umap[mask, 2],
                mode="markers",
                name=f"故障编码：{cls_name}",
                marker=dict(
                    size=point_size,
                    color=colors[i % len(colors)],
                    opacity=opacity,
                    line=dict(width=0.5, color="black"),
                ),
                text=[cls_name] * np.sum(mask),
                hovertemplate=(
                    "类别: %{text}<br>"
                    "UMAP特征1: %{x:.3f}<br>"
                    "UMAP特征2: %{y:.3f}<br>"
                    "UMAP特征3: %{z:.3f}<extra></extra>"
                ),
            )
        )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="UMAP特征1",
            yaxis_title="UMAP特征2",
            zaxis_title="UMAP特征3",
            bgcolor="white",
            xaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
            aspectmode="data",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=0, r=0, t=50, b=60),
        template="plotly_white",
    )

    save_path = os.fspath(save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plot(fig, filename=save_path, auto_open=False, include_plotlyjs=True)

    return X_umap

def plot_pacmap_3d_html(
    X,
    y,
    unique_classes=("N", "L1", "L2", "L3", "L2L3"),
    save_path=None,
    random_state=36,
    n_neighbors=10,
    MN_ratio=0.5,
    FP_ratio=2.0,
    point_size=5,
    opacity=0.9,
    title="3D PaCMAP Visualization"
):
    """
    生成可交互旋转的 3D PaCMAP HTML 文件

    参数
    ----
    X : np.ndarray, shape [N, D]
        输入特征
    y : np.ndarray or list, shape [N]
        标签，可为字符串或整数
    unique_classes : tuple/list
        类别顺序
    save_path : str or Path
        输出 html 文件路径
    random_state : int
        随机种子
    n_neighbors : int
        邻居数
    MN_ratio : float
        Mid-near pair ratio
    FP_ratio : float
        Further pair ratio
    point_size : int/float
        散点大小
    opacity : float
        散点透明度
    title : str
        图标题
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X 应为二维数组 [N, D]，当前形状为 {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X 和 y 长度不一致: len(X)={len(X)}, len(y)={len(y)}")
    if len(X) < 3:
        raise ValueError("样本数过少，至少需要 3 个样本")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X 中存在 NaN 或 Inf，请先清理数据")

    # 标签映射
    if y.dtype.kind in {"U", "S", "O"}:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        try:
            y_idx = np.array([class_to_idx[label] for label in y])
        except KeyError as e:
            raise ValueError(f"y 中存在未在 unique_classes 中定义的标签: {e}")
    else:
        y_idx = y.astype(int)

    reducer = pacmap.PaCMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        random_state=random_state
    )
    X_pacmap = reducer.fit_transform(X)

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]

    traces = []
    for i, cls_name in enumerate(unique_classes):
        mask = (y_idx == i)
        if np.sum(mask) == 0:
            continue

        traces.append(
            go.Scatter3d(
                x=X_pacmap[mask, 0],
                y=X_pacmap[mask, 1],
                z=X_pacmap[mask, 2],
                mode="markers",
                name=f"故障编码：{cls_name}",
                marker=dict(
                    size=point_size,
                    color=colors[i % len(colors)],
                    opacity=opacity,
                    line=dict(width=0.5, color="black"),
                ),
                text=[cls_name] * np.sum(mask),
                hovertemplate=(
                    "类别: %{text}<br>"
                    "PaCMAP特征1: %{x:.3f}<br>"
                    "PaCMAP特征2: %{y:.3f}<br>"
                    "PaCMAP特征3: %{z:.3f}<extra></extra>"
                ),
            )
        )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title="PaCMAP特征1",
            yaxis_title="PaCMAP特征2",
            zaxis_title="PaCMAP特征3",
            bgcolor="white",
            xaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
            yaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
            zaxis=dict(backgroundcolor="white", gridcolor="lightgray", zerolinecolor="lightgray"),
            aspectmode="data",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=0, r=0, t=50, b=60),
        template="plotly_white",
    )

    save_path = os.fspath(save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plot(fig, filename=save_path, auto_open=False, include_plotlyjs=True)

    return X_pacmap

def plot_pacmap_2d_html(
    X,
    y,
    unique_classes=("N", "L1", "L2", "L3", "L2L3"),
    save_path=None,
    random_state=36,
    n_neighbors=10,
    MN_ratio=0.5,
    FP_ratio=2.0,
    point_size=7,
    opacity=0.9,
    title="2维 PaCMAP 可视化"
):
    """
    生成可交互缩放/平移的 2D PaCMAP HTML 文件

    参数
    ----
    X : np.ndarray, shape [N, D]
        输入特征
    y : np.ndarray or list, shape [N]
        标签，可为字符串或整数
    unique_classes : tuple/list
        类别顺序
    save_path : str or Path
        输出 html 文件路径
    random_state : int
        随机种子
    n_neighbors : int
        邻居数
    MN_ratio : float
        Mid-near pair ratio
    FP_ratio : float
        Further pair ratio
    point_size : int/float
        散点大小
    opacity : float
        散点透明度
    title : str
        图标题
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X 应为二维数组 [N, D]，当前形状为 {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"X 和 y 长度不一致: len(X)={len(X)}, len(y)={len(y)}")
    if len(X) < 3:
        raise ValueError("样本数过少，至少需要 3 个样本")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("X 中存在 NaN 或 Inf，请先清理数据")

    if y.dtype.kind in {"U", "S", "O"}:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}
        try:
            y_idx = np.array([class_to_idx[label] for label in y])
        except KeyError as e:
            raise ValueError(f"y 中存在未在 unique_classes 中定义的标签: {e}")
    else:
        y_idx = y.astype(int)

    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        random_state=random_state
    )
    X_pacmap = reducer.fit_transform(X)

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]

    traces = []
    for i, cls_name in enumerate(unique_classes):
        mask = (y_idx == i)
        if np.sum(mask) == 0:
            continue

        traces.append(
            go.Scatter(
                x=X_pacmap[mask, 0],
                y=X_pacmap[mask, 1],
                mode="markers",
                name=f"故障编码：{cls_name}",
                marker=dict(
                    size=point_size,
                    color=colors[i % len(colors)],
                    opacity=opacity,
                    line=dict(width=0.5, color="black"),
                ),
                text=[cls_name] * np.sum(mask),
                hovertemplate=(
                    "类别: %{text}<br>"
                    "PaCMAP特征1: %{x:.3f}<br>"
                    "PaCMAP特征2: %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="PaCMAP特征1",
        yaxis_title="PaCMAP特征2",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=50, r=30, t=50, b=80),
        template="plotly_white",
    )

    save_path = os.fspath(save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plot(fig, filename=save_path, auto_open=False, include_plotlyjs=True)

    return X_pacmap

def plot_confusion_matrix(cm, class_names, save_path, font_path, figsize, dpi):
    """
    绘制混淆矩阵

    参数
    ----
    cm : np.ndarray, shape [C, C]
        混淆矩阵
    class_names : list of str
        类别名称列表，顺序应与 cm 的行列顺序一致
    save_path : str
        图片保存路径
    font_path : str
        Linux 下中文字体路径
    figsize : tuple
        图像尺寸
    dpi : int
        保存分辨率
    """

    # ========= 1. 检查字体 =========
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件不存在: {font_path}")

    font_prop = font_manager.FontProperties(fname=font_path)

    # 注册字体，避免某些 Linux 环境下中文失效
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

    # ========= 2. 绘制混淆矩阵 =========
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("样本数量", rotation=-90, va="bottom", fontproperties=font_prop)

    # 设置坐标轴标签和标题
    ax.set_xlabel("预测标签", fontproperties=font_prop, fontsize=13)
    ax.set_ylabel("真实标签", fontproperties=font_prop, fontsize=13)

    # 设置坐标轴刻度和标签
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontproperties=font_prop, fontsize=11)
    ax.set_yticklabels(class_names, fontproperties=font_prop, fontsize=11)

    # 在每个格子中添加数字标签
    thresh = cm.max() / 2.0  # 阈值，用于决定文本颜色
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color=color,
                fontproperties=font_prop,
                fontsize=10,
            )
    # ========= 3. 保存 =========
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    

def plot_confusion_matrix_fused(
    logits,
    targets,
    save_path,
    font_path="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    figsize=(7.2, 6.4),
    dpi=400,
):
    """
    按照融合标签逻辑绘制混淆矩阵
    不修改原 plot_confusion_matrix 的任何配置
    """
    if isinstance(logits, torch.Tensor):
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
    else:
        logits = np.asarray(logits)
        preds = np.argmax(logits, axis=1)

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    fused_preds = fuse_labels(preds)
    fused_targets = fuse_labels(targets)

    fused_class_names = ["N", "L1", "L2/L3/L2L3"]

    cm = confusion_matrix(
        fused_targets,
        fused_preds,
        labels=[0, 1, 2]
    )

    plot_confusion_matrix(
        cm=cm,
        class_names=fused_class_names,
        save_path=save_path,
        font_path=font_path,
        figsize=figsize,
        dpi=dpi,
    )

    return cm

def plot_history_metrics_separate(history_json_path, save_dir):
    """
    读取 history.json，并分别绘制：
    1) 训练集 loss 类指标
    2) 训练集 acc 类指标
    3) 验证集 loss 类指标
    4) 验证集 acc 类指标

    参数
    ----
    history_json_path : str
        history.json 文件路径
    save_dir : str
        图片保存目录
    """
    loss_metrics = ["loss", "ce_loss", "l_in", "l_out", "l_domain"]
    acc_metrics = ["acc", "merged_acc"]

    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, list) or len(history) == 0:
        raise ValueError("history.json 内容格式错误，应为非空列表。")

    os.makedirs(save_dir, exist_ok=True)

    epochs = [item.get("epoch", idx + 1) for idx, item in enumerate(history)]

    train_data = {m: [] for m in loss_metrics + acc_metrics}
    val_data = {m: [] for m in loss_metrics + acc_metrics}

    for item in history:
        train_part = item.get("train", {})
        val_part = item.get("val", {})

        for m in loss_metrics + acc_metrics:
            train_data[m].append(train_part.get(m, None))
            val_data[m].append(val_part.get(m, None))

    def _plot_one_group(metric_names, data_dict, title, save_path):
        plt.figure(figsize=(10, 6))
        for metric in metric_names:
            plt.plot(epochs, data_dict[metric], label=metric, linewidth=2)

        plt.title(title, fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    # 1. train loss
    _plot_one_group(
        metric_names=loss_metrics,
        data_dict=train_data,
        title="Train Loss Metrics",
        save_path=os.path.join(save_dir, "train_loss.png")
    )

    # 2. train acc
    _plot_one_group(
        metric_names=acc_metrics,
        data_dict=train_data,
        title="Train Accuracy Metrics",
        save_path=os.path.join(save_dir, "train_acc.png")
    )

    # 3. val loss
    _plot_one_group(
        metric_names=loss_metrics,
        data_dict=val_data,
        title="Validation Loss Metrics",
        save_path=os.path.join(save_dir, "val_loss.png")
    )

    # 4. val acc
    _plot_one_group(
        metric_names=acc_metrics,
        data_dict=val_data,
        title="Validation Accuracy Metrics",
        save_path=os.path.join(save_dir, "val_acc.png")
    )

    print(f"图片已保存到目录: {save_dir}")


if __name__ == "__main__":
    # =========================
    # 示例：用模拟数据演示
    # 你替换成自己的特征和标签即可
    # =========================
    np.random.seed(42)

    unique_classes = ["N", "L2", "L3", "L2L3", "L1"]

    num_per_class = 30
    feat_dim = 128

    X_list = []
    y_list = []

    for i, cls in enumerate(unique_classes):
        center = np.random.randn(feat_dim) * 3 + i * 2
        samples = center + np.random.randn(num_per_class, feat_dim) * 0.8
        X_list.append(samples)
        y_list.extend([cls] * num_per_class)
    
    X = np.vstack(X_list)   # shape [N, D]
    y = np.array(y_list)    # shape [N]

    plot_tsne(
        X=X,
        y=y,
        unique_classes=unique_classes,
        save_path="tsne_plot.jpg",
        font_path="/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        figsize=(7.2, 6.4),
        dpi=400,
        random_state=42,
        perplexity=30,
        n_iter=1000,
        point_size=10,
        alpha=0.9,
    )