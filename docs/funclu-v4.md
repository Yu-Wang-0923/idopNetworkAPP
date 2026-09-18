# FunClu v4 内核迁移

拟合与聚类数值实现来自用户提供的 `funclu_v4(3).py`。独立脚本中的本机路径、自动运行与绘图入口未引入应用。

- `power_fitting` 使用标准化 log(time) 的向量化回归，返回 `c、beta、a、b、n_positive`；默认生成 30 个等距时间采样点。
- `clustering/_v4.py` 保存 v4 求解器：参数空间 KMeans 初始化、SAD 均值优化、每个条件共享协方差、条件权重平衡。保留原生 `select_best_k` 接口。
- `clustering/funclu.py` 是应用兼容层，现有 `fit(data_list)`、tensor 结果、`get_cluster_curves` 和 ZIP 导出继续可用。可用 `parameter_data` 传入各条件的原始拟合参数；未提供时从输入曲线拟合参数。
- 页面从 ZIP 内的 quasi_dynamic 数据计算初始化参数，兼容旧 ZIP。原始标准化参数在 `model.kernel_.params_mu_`，对外 `params_mu` 转回 `a、b`，避免旧页面和网络重建误读。
- 新 BIC 参数数为 `(K - 1) + 2*K*L + 2*L`。页面保留原有多随机种子汇总和 K 推荐方式，数值计算使用 v4；默认最大迭代次数升至 1000。
- 数据变换器同样使用 v4 的 `preprocess`：`None`、`Log10_1p`、`Minmax_0_1`、`Z_min_add1`。非数值单元格转为 NaN；变换按列进行，`Z_min_add1` 为每列减最小值再加 1，不执行 z-score 标准化。Curve Fitting 页面点击 Run Fitting 后执行新脚本默认流程：`Z_min_add1 → Log10_1p → log1p(行和)` 排序，保留正时间并去除前 1% 后拟合。

边界处理：时间轴需至少两个不同的有限正值；无足够正数观测的特征参数保留 NaN，由聚类器排除；K 不得超过有效共同特征数；空初始化簇使用有限均值兜底；非有限聚类结果报错。BIC 表保留各次拟合错误信息。

建议重新运行拟合器并生成 ZIP，以使用新版的 30 点采样。旧版会话中的聚类结果需要重新拟合。

测试：

```sh
PYTHONPATH=packages/idopnetwork/src:packages/idopnetwork-app/src python -m unittest discover -s tests -v
```

页面提供两步变换、采样点数、去除行百分比设置，点击 Run Fitting 后统一执行并显示结果。缓存同时包含文件内容和参数，上传文件改变后清除旧结果；图形调整不会触发拟合。移除 Dynamic Data 和分步执行入口；保留原有绘图函数、绘图设置及多文件对比。ZIP 保留原有三表路径以兼容 FunClu 和 NetRecon，单表下载按新脚本的 params/samples/quasidynamic 命名。
