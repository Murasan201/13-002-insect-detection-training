# セットアップガイド（技術書原稿用）

<!--
本ドキュメントは技術書の原稿として使用するためのセットアップガイドです。
読者がゼロから環境を構築し、本プロジェクトのスクリプトを実行できるよう、
詳細かつ漏れのないインストール手順を記載します。
-->

## 本ガイドの目的

本ガイドでは、YOLOv8を使用した昆虫（カブトムシ）検出モデルの学習環境を構築する手順を解説します。Python仮想環境の作成から、必要なライブラリのインストール、データセットの準備、学習スクリプトの実行までを、実際のコマンドと出力例を交えて詳細に説明します。

## 動作環境

本ガイドではLinux環境を前提に解説しますが、macOSでもCPU処理であれば同じ手順で実行できます。

## セットアップ手順

### 1. システム要件の確認

本プロジェクトを実行するには、以下の環境が必要です：

- **OS**: Linux（Ubuntu 20.04以降推奨）またはmacOS
- **CPU**: x86_64アーキテクチャ
- **メモリ**: 8GB以上（16GB推奨）
- **ディスク空き容量**: 10GB以上（PyTorchとデータセット用）
- **ネットワーク**: インターネット接続（ライブラリダウンロード用）

### 2. Python環境の準備

Python 3.9以上がインストールされていることを確認します：

```bash
python3 --version
```

出力例：
```
Python 3.10.12
```

Pythonがインストールされていない場合は、以下のコマンドでインストールします：

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

### 3. 仮想環境の作成

プロジェクト専用のPython仮想環境を作成します。仮想環境を使用することで、システムのPython環境を汚さずに必要なライブラリをインストールできます。

```bash
# プロジェクトディレクトリに移動
cd 13-002-insect-detection-training

# 仮想環境を作成
python3 -m venv venv
```

#### 3.1 仮想環境の有効化

仮想環境を有効化します。有効化後はプロンプトの先頭に`(venv)`が表示されます。

```bash
source venv/bin/activate
```

有効化されたことを確認します：

```bash
python --version
pip --version
```

出力：
```
Python 3.10.12
pip 22.0.2 from /home/win/work/projetct/kodansya/install-test/13-002-insect-detection-training/venv/lib/python3.10/site-packages/pip (python 3.10)
```

### 4. 依存ライブラリのインストール

学習スクリプトを実行するために必要なライブラリを1つずつインストールします。

#### 4.1 pipのアップグレード

最新版のpipにアップグレードします：

```bash
pip install --upgrade pip
```

出力：
```
Requirement already satisfied: pip in ./venv/lib/python3.10/site-packages (22.0.2)
Collecting pip
  Using cached pip-25.3-py3-none-any.whl (1.8 MB)
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 22.0.2
    Uninstalling pip-22.0.2:
      Successfully uninstalled pip-22.0.2
Successfully installed pip-25.3
```

#### 4.2 PyTorchのインストール

深層学習フレームワークであるPyTorchをインストールします。PyTorchは本プロジェクトの基盤となるライブラリです。

```bash
pip install torch
```

出力：
```
Collecting torch
  Using cached torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (30 kB)
Collecting filelock (from torch)
  Using cached filelock-3.20.1-py3-none-any.whl.metadata (2.1 kB)
Collecting typing-extensions>=4.10.0 (from torch)
  Using cached typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
Collecting sympy>=1.13.3 (from torch)
  Using cached sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Collecting networkx>=2.5.1 (from torch)
  Using cached networkx-3.4.2-py3-none-any.whl.metadata (6.3 kB)
Collecting jinja2 (from torch)
  Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting fsspec>=0.8.5 (from torch)
  Using cached fsspec-2025.12.0-py3-none-any.whl.metadata (10 kB)
Collecting nvidia-cuda-nvrtc-cu12==12.8.93 (from torch)
  ...（中略）...
Using cached torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl (899.8 MB)
  ...（中略）...
Installing collected packages: nvidia-cusparselt-cu12, mpmath, typing-extensions, triton, sympy, nvidia-nvtx-cu12, nvidia-nvshmem-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufile-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, networkx, MarkupSafe, fsspec, filelock, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, jinja2, nvidia-cusolver-cu12, torch

Successfully installed MarkupSafe-3.0.3 filelock-3.20.1 fsspec-2025.12.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.4.2 nvidia-cublas-cu12-12.8.4.1 nvidia-cuda-cupti-cu12-12.8.90 nvidia-cuda-nvrtc-cu12-12.8.93 nvidia-cuda-runtime-cu12-12.8.90 nvidia-cudnn-cu12-9.10.2.21 nvidia-cufft-cu12-11.3.3.83 nvidia-cufile-cu12-1.13.1.3 nvidia-curand-cu12-10.3.9.90 nvidia-cusolver-cu12-11.7.3.90 nvidia-cusparse-cu12-12.5.8.93 nvidia-cusparselt-cu12-0.7.1 nvidia-nccl-cu12-2.27.5 nvidia-nvjitlink-cu12-12.8.93 nvidia-nvshmem-cu12-3.3.20 nvidia-nvtx-cu12-12.8.90 sympy-1.14.0 torch-2.9.1 triton-3.5.1 typing-extensions-4.15.0
```

**注意:** PyTorchのダウンロードには時間がかかります（約900MB）。安定したネットワーク環境で実行してください。

#### 4.3 Ultralyticsのインストール

YOLOv8の実装を提供するUltralyticsライブラリをインストールします。このライブラリにより、物体検出モデルの学習と推論が可能になります。

```bash
pip install ultralytics
```

出力：
```
Collecting ultralytics
  Using cached ultralytics-8.3.241-py3-none-any.whl.metadata (37 kB)
Collecting numpy>=1.23.0 (from ultralytics)
  Using cached numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
Collecting matplotlib>=3.3.0 (from ultralytics)
  Using cached matplotlib-3.10.8-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (52 kB)
Collecting opencv-python>=4.6.0 (from ultralytics)
  Using cached opencv_python-4.12.0.88-cp37-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (19 kB)
Collecting pillow>=7.1.2 (from ultralytics)
  Using cached pillow-12.0.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.8 kB)
Collecting pyyaml>=5.3.1 (from ultralytics)
  Using cached pyyaml-6.0.3-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.4 kB)
Collecting requests>=2.23.0 (from ultralytics)
  Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Collecting scipy>=1.4.1 (from ultralytics)
  Using cached scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
Requirement already satisfied: torch>=1.8.0 in ./venv/lib/python3.10/site-packages (from ultralytics) (2.9.1)
Collecting torchvision>=0.9.0 (from ultralytics)
  Using cached torchvision-0.24.1-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (5.9 kB)
Collecting psutil>=5.8.0 (from ultralytics)
  Using cached psutil-7.2.0-cp36-abi3-manylinux2010_x86_64.manylinux_2_12_x86_64.manylinux_2_28_x86_64.whl.metadata (22 kB)
Collecting polars>=0.20.0 (from ultralytics)
  Using cached polars-1.36.1-py3-none-any.whl.metadata (10 kB)
Collecting ultralytics-thop>=2.0.18 (from ultralytics)
  Using cached ultralytics_thop-2.0.18-py3-none-any.whl.metadata (14 kB)
  ...（中略）...
Installing collected packages: urllib3, six, pyyaml, pyparsing, psutil, polars-runtime-32, pillow, packaging, numpy, kiwisolver, idna, fonttools, cycler, charset_normalizer, certifi, scipy, requests, python-dateutil, polars, opencv-python, contourpy, matplotlib, ultralytics-thop, torchvision, ultralytics

Successfully installed certifi-2025.11.12 charset_normalizer-3.4.4 contourpy-1.3.2 cycler-0.12.1 fonttools-4.61.1 idna-3.11 kiwisolver-1.4.9 matplotlib-3.10.8 numpy-2.2.6 opencv-python-4.12.0.88 packaging-25.0 pillow-12.0.0 polars-1.36.1 polars-runtime-32-1.36.1 psutil-7.2.0 pyparsing-3.3.1 python-dateutil-2.9.0.post0 pyyaml-6.0.3 requests-2.32.5 scipy-1.15.3 six-1.17.0 torchvision-0.24.1 ultralytics-8.3.241 ultralytics-thop-2.0.18 urllib3-2.6.2
```

Ultralyticsをインストールすると、依存関係として以下のライブラリも自動的にインストールされます：

| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| numpy | 2.2.6 | 数値計算 |
| opencv-python | 4.12.0.88 | 画像処理 |
| pillow | 12.0.0 | 画像処理 |
| matplotlib | 3.10.8 | グラフ描画 |
| scipy | 1.15.3 | 科学技術計算 |
| torchvision | 0.24.1 | PyTorchの画像処理拡張 |
| pyyaml | 6.0.3 | YAML設定ファイル読み込み |

#### 4.4 インストールの確認

ライブラリが正しくインストールされたことを確認します：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

出力：
```
PyTorch: 2.9.1+cu128
Ultralytics: OK
OpenCV: 4.12.0
```

これで必要なライブラリのインストールは完了です。

### 5. データセットの準備

本プロジェクトでは、昆虫（カブトムシ）検出モデルの学習にRoboflowで公開されているデータセットを使用します。

#### 5.1 データセットのダウンロード

1. 以下のURLにアクセスします：
   - https://universe.roboflow.com/z-algae-bilby/beetle/dataset/1

2. Roboflowのアカウントでログインします（無料アカウントで利用可能）

3. 「Download Dataset」ボタンをクリックします

4. フォーマット選択画面で以下を選択します：
   - **Format**: `YOLOv8`
   - **Download Type**: `zip`

5. 「Continue」をクリックしてZIPファイルをダウンロードします
   - ファイル名例: `Beetle.v1i.yolov8.zip`
   - ファイルサイズ: 約45MB

#### 5.2 データセットの配置

ダウンロードしたZIPファイルをプロジェクトの`downloads/`ディレクトリに配置します。

```bash
# プロジェクトディレクトリに移動
cd 13-002-insect-detection-training

# downloads/ディレクトリが存在しない場合は作成
mkdir -p downloads

# ダウンロードしたZIPファイルを移動
# （ダウンロード先がDownloadsフォルダの場合）
mv ~/Downloads/Beetle.v1i.yolov8.zip downloads/
```

#### 5.3 配置の確認

ZIPファイルが正しく配置されたことを確認します。

```bash
ls -la downloads/
```

以下のように表示されれば成功です：

```
downloads/
└── Beetle.v1i.yolov8.zip  (約45MB)
```

#### 5.4 データセットの展開

`setup_dataset.py`スクリプトを使用してデータセットを展開します。このスクリプトはPython標準ライブラリのみを使用しているため、追加のライブラリインストールは不要です。

```bash
python3 setup_dataset.py
```

実行すると以下のような出力が表示されます：

```
============================================================
カブトムシ検出データセット セットアップ
============================================================

データセット情報:
   提供元: Roboflow Universe (z-algae-bilby)
   ライセンス: CC BY 4.0
   URL: https://universe.roboflow.com/z-algae-bilby/beetle/dataset/1

ZIPファイルを発見: Beetle.v1i.yolov8.zip

ZIPファイルを展開中: Beetle.v1i.yolov8.zip
   展開先: datasets

   1012 個のファイルを展開します...
展開完了

データセット構造を検証中...

data.yaml: 見つかりました
train/images: 400 ファイル (訓練用画像)
train/labels: 400 ファイル (訓練用ラベル（YOLO形式のtxtファイル）)
valid/images: 50 ファイル (検証用画像)
valid/labels: 50 ファイル (検証用ラベル)
test/images: 50 ファイル (テスト用画像・オプション)

データセット構造の検証: 成功

データセット統計:
   訓練画像: 400 枚
   検証画像: 50 枚
   合計: 450 枚

ZIPファイルを保持: downloads/Beetle.v1i.yolov8.zip

データセットの準備が完了しました
```

#### 5.5 展開結果の確認

展開後、`datasets/`ディレクトリに以下の構造が作成されていることを確認します。

```bash
ls -la datasets/
```

出力例：

```
datasets/
├── README.dataset.txt    # データセット情報
├── README.roboflow.txt   # Roboflow情報
├── data.yaml             # YOLOv8設定ファイル
├── train/                # 訓練用データ
│   ├── images/           # 400枚の画像
│   └── labels/           # 400個のラベルファイル
├── valid/                # 検証用データ
│   ├── images/           # 50枚の画像
│   └── labels/           # 50個のラベルファイル
└── test/                 # テスト用データ（オプション）
    ├── images/           # 50枚の画像
    └── labels/           # 50個のラベルファイル
```

**データセット統計:**
| 種別 | 画像数 | 用途 |
|------|--------|------|
| train | 400枚 | モデルの学習 |
| valid | 50枚 | 学習中の検証 |
| test | 50枚 | 最終評価（オプション） |
| **合計** | **500枚** | |

これでデータセットの準備は完了です。

### 6. モデルファイルの取得

学習スクリプトを実行すると、YOLOv8の事前学習モデル（yolov8n.pt）が自動的にダウンロードされます。手動でダウンロードする必要はありません。

初回実行時に以下のようなメッセージが表示されます：

```
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt to 'yolov8n.pt'...
100% ━━━━━━━━━━━━ 6.2MB 4.7MB/s
```

**学習済みモデルについて：**

本プロジェクトで学習したカブトムシ検出モデルは、Hugging Faceで公開されています：
- URL: https://huggingface.co/Murasan/beetle-detection-yolov8
- フォーマット: PyTorch (.pt)、ONNX (.onnx)

### 7. 動作確認

セットアップが正しく完了したことを確認するため、学習スクリプトを実行します。

#### 7.1 学習スクリプトの実行

仮想環境が有効化されていることを確認し、学習スクリプトを実行します：

```bash
# 仮想環境が有効化されていない場合は有効化
source venv/bin/activate

# 学習スクリプトを実行（テスト用に1エポックのみ）
python train_yolo.py --data datasets/data.yaml --epochs 1 --batch 8 --device cpu
```

#### 7.2 実行時の出力

学習が正常に開始されると、以下のような出力が表示されます：

```
============================================================
🐛 YOLOv8 昆虫検出モデル訓練スクリプト
============================================================
Pythonバージョン: 3.10.12
PyTorchバージョン: 2.9.1+cu128
CUDA利用可能: False
検出されたGPU数: 0
訓練はCPUのみで実行されます（GPUよりも遅くなります）
OpenCVバージョン: 4.12.0
train/images に 400 個のファイルを発見
train/labels に 400 個のファイルを発見
valid/images に 50 個のファイルを発見
valid/labels に 50 個のファイルを発見
データセットの検証が成功しました
YOLOv8訓練プロセスを開始します
モデル: yolov8n.pt
データセット: datasets/data.yaml
エポック数: 1
バッチサイズ: 8
画像サイズ: 640
デバイス: cpu
```

初回実行時は事前訓練モデル（yolov8n.pt、約6MB）が自動的にダウンロードされます。

#### 7.3 学習の進行

学習が進行すると、エポックごとの進捗が表示されます：

```
Ultralytics 8.3.241 🚀 Python-3.10.12 torch-2.9.1+cu128 CPU
Model summary: 129 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
        1/1         0G      1.403      2.109      1.587         21        640: 100%
```

#### 7.4 学習完了の確認

学習が正常に完了すると、以下のメッセージが表示されます：

```
訓練が 260.31 秒（4.3 分）で完了しました
訓練結果とモデルの保存先: training_results/beetle_detection
mAP@0.5: 0.5237 (IoU閾値0.5での平均精度)
mAP@0.5:0.95: 0.2580 (IoU閾値0.5-0.95での平均精度)
Precision: 0.9256 (精度: 正しい検出の割合)
Recall: 0.1818 (再現率: 実際のオブジェクトを検出できた割合)
🎉 訓練パイプラインが成功しました！
💾 モデル重みの保存先: training_results/beetle_detection/weights/
```

**注意:** 1エポックのみの学習では精度が低いため、実際の運用では100エポック程度の学習を推奨します。

#### 7.5 学習結果の確認

学習結果は`training_results/beetle_detection/`ディレクトリに保存されます：

```bash
ls -la training_results/beetle_detection/
```

出力例：
```
training_results/beetle_detection/
├── weights/
│   ├── best.pt     # 最も精度の高いモデル
│   └── last.pt     # 最後のエポックのモデル
├── results.csv     # 学習結果のログ
├── results.png     # 学習曲線のグラフ
├── confusion_matrix.png  # 混同行列
└── ...
```

これでセットアップと動作確認は完了です。

## トラブルシューティング

### pip install でエラーが発生する場合

pipのバージョンが古い可能性があります。以下のコマンドでアップグレードしてください：

```bash
pip install --upgrade pip
```

### メモリ不足エラーが発生する場合

バッチサイズを小さくして実行してください：

```bash
python train_yolo.py --data datasets/data.yaml --epochs 1 --batch 4 --device cpu
```

### 仮想環境が有効化されていない場合

コマンド実行前に仮想環境を有効化してください：

```bash
source venv/bin/activate
```

プロンプトの先頭に`(venv)`が表示されていることを確認してください。

## 次のステップ

セットアップが完了したら、以下の操作が可能です：

1. **本格的な学習の実行**: エポック数を増やして精度の高いモデルを学習
   ```bash
   python train_yolo.py --data datasets/data.yaml --epochs 100 --batch 16
   ```

2. **推論の実行**: 学習済みモデルで画像から昆虫を検出
   ```bash
   python detect_insect.py --input input_images/ --output output_images/
   ```

3. **モデルのエクスポート**: ONNX形式への変換など
