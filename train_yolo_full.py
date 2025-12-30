#!/usr/bin/env python3
"""
YOLOv8 昆虫検出モデル訓練スクリプト

このスクリプトは、カスタム昆虫データセットを使用してYOLOv8モデルのファインチューニングを実行します。
Roboflowデータセットを使用したカブトムシ検出に特化して設計されています。

要件定義書: docs/insect_detection_application_test_project_requirements_spec.md

使用方法:
    python train_yolo.py --data datasets/data.yaml --epochs 100
    python train_yolo.py --data datasets/data.yaml --epochs 50 --batch 16 --imgsz 640

必要なライブラリ:
    - ultralytics
    - torch
    - opencv-python
    - numpy
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 機械学習とコンピュータビジョンに必要なライブラリのインポート
# インポートエラーでスクリプトが停止しないようtry-exceptでラップ
try:
    from ultralytics import YOLO  # Ultralytics製 YOLOv8モデルライブラリ
    import torch                 # PyTorch深層学習フレームワーク
    import cv2                   # OpenCVコンピュータビジョンライブラリ
    import numpy as np           # NumPy数値計算ライブラリ
except ImportError as e:
    # ライブラリがインストールされていない場合のエラーハンドリング
    print(f"エラー: 必要なライブラリがインストールされていません: {e}")
    print("依存関係をインストールしてください: pip install -r requirements.txt")
    sys.exit(1)


def setup_logging():
    """
    訓練プロセス用のログ設定を初期化します。

    Returns:
        logging.Logger: 設定済みのロガーインスタンス
    """
    # 現在の日時を使用してユニークなログファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)  # logsディレクトリが存在しない場合は自動作成
    
    # 訓練セッション固有のログファイルパスを作成
    log_file = log_dir / f"training_{timestamp}.log"
    
    # ログシステムの基本設定（レベル、フォーマット、出力先）
    logging.basicConfig(
        level=logging.INFO,                    # 情報レベル以上をログ出力
        format='%(asctime)s - %(levelname)s - %(message)s',  # タイムスタンプ付きフォーマット
        handlers=[
            logging.FileHandler(log_file),    # ログファイルへの出力
            logging.StreamHandler(sys.stdout)  # コンソールへの同時出力
        ]
    )
    
    # このモジュール用のロガーインスタンスを返す
    return logging.getLogger(__name__)


def validate_dataset(data_path):
    """
    データセットの構造と設定を検証します。
    
    Args:
        data_path (str): data.yamlファイルのパス
        
    Returns:
        bool: データセットが有効な場合True、そうでなければFalse
    """
    data_file = Path(data_path)
    if not data_file.exists():
        logging.error(f"データセット設定ファイルが見つかりません: {data_path}")
        return False
    
    # データセットの標準的なYOLO形式ディレクトリ構造を確認
    dataset_dir = data_file.parent
    # YOLOデータセットに必要なディレクトリのリスト
    required_dirs = ["train/images", "train/labels", "valid/images", "valid/labels"]
    
    # 各ディレクトリの存在とファイル数をチェック
    for dir_path in required_dirs:
        full_path = dataset_dir / dir_path
        if not full_path.exists():
            logging.error(f"必要なディレクトリが見つかりません: {full_path}")
            return False
        
        # ディレクトリ内のすべてのファイルを取得して数をカウント
        files = list(full_path.glob("*"))
        if not files:
            # 空のディレクトリは訓練に使用できない
            logging.error(f"ディレクトリ内にファイルがありません: {full_path}")
            return False
        
        # データセットのサイズ情報をログ出力
        logging.info(f"{dir_path} に {len(files)} 個のファイルを発見")
    
    logging.info("データセットの検証が成功しました")
    return True


def check_system_requirements():
    """システム要件をチェックし、システム情報をログに記録します。"""
    logger = logging.getLogger(__name__)
    
    # 実行環境のPythonバージョンをチェック（互換性確認のため）
    python_version = sys.version
    logger.info(f"Pythonバージョン: {python_version}")
    
    # PyTorchのバージョンとGPUサポート状況を確認
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()  # CUDA（NVIDIA GPU）が使用可能かチェック
    device_count = torch.cuda.device_count() if cuda_available else 0  # 使用可能GPU数
    
    logger.info(f"PyTorchバージョン: {torch_version}")
    logger.info(f"CUDA利用可能: {cuda_available}")
    logger.info(f"検出されたGPU数: {device_count}")
    
    if cuda_available:
        # 各GPUの詳細情報（モデル名、メモリ容量など）を表示
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            logger.info(f"GPU {i}: {gpu_name}")
    else:
        # GPUが使用できない場合の警告（訓練時間が大幅に延長する可能性）
        logger.info("訓練はCPUのみで実行されます（GPUよりも遅くなります）")
    
    # OpenCVバージョンをチェック（画像処理機能の確認）
    cv2_version = cv2.__version__
    logger.info(f"OpenCVバージョン: {cv2_version}")


def train_model(data_path, model_name="yolov8n.pt", epochs=100, batch_size=16, 
                img_size=640, device="auto", project="training_results", 
                name="beetle_detection"):
    """
    指定されたパラメータでYOLOv8モデルを訓練します。
    
    Args:
        data_path (str): データセット設定ファイルのパス
        model_name (str): 使用する事前訓練モデル
        epochs (int): 訓練エポック数
        batch_size (int): 訓練時のバッチサイズ
        img_size (int): 訓練用画像サイズ
        device (str): 訓練に使用するデバイス
        project (str): プロジェクトディレクトリ名
        name (str): 実験名
        
    Returns:
        YOLO: 訓練済みモデルインスタンス
    """
    logger = logging.getLogger(__name__)
    
    logger.info("YOLOv8訓練プロセスを開始します")
    logger.info(f"モデル: {model_name}")
    logger.info(f"データセット: {data_path}")
    logger.info(f"エポック数: {epochs}")
    logger.info(f"バッチサイズ: {batch_size}")
    logger.info(f"画像サイズ: {img_size}")
    logger.info(f"デバイス: {device}")
    
    try:
        # COCOデータセットで事前訓練されたYOLOv8モデルをベースとして読み込み
        logger.info(f"事前訓練モデルを読み込み中: {model_name}")
        model = YOLO(model_name)  # モデルインスタンスを作成
        
        # 訓練処理時間の計測を開始
        start_time = time.time()
        logger.info("カスタムデータセットでのファインチューニングを開始します...")
        
        # YOLOv8の訓練メソッドを実行（ファインチューニング）
        results = model.train(
            data=data_path,      # データセットの設定ファイル（data.yaml）
            epochs=epochs,       # 全データセットを何回繰り返すか
            batch=batch_size,    # 1回の更新で使用する画像数
            imgsz=img_size,      # 訓練時の画像リサイズ（正方形）
            device=device,       # 計算デバイス（auto、cpu、0、1など）
            project=project,     # 訓練結果を保存するプロジェクトディレクトリ
            name=name,           # この訓練セッションの実験名
            save=True,           # モデル重みの保存を有効化
            save_period=10,      # 指定エポック数ごとに中間チェックポイントを保存
            val=True,            # 訓練中の検証データでの性能評価を有効化
            plots=True,          # 訓練進行と結果のグラフ出力を有効化
            verbose=True         # 訓練中の詳細なログ情報を表示
        )
        
        # 訓練完了時間を計算してログ出力
        training_time = time.time() - start_time
        logger.info(f"訓練が {training_time:.2f} 秒（{training_time/60:.1f} 分）で完了しました")
        logger.info(f"訓練結果とモデルの保存先: {project}/{name}")
        
        # 訓練済みモデルと訓練結果を返す
        return model, results
        
    except Exception as e:
        logger.error(f"訓練が失敗しました: {str(e)}")
        raise


def validate_model(model, data_path):
    """
    訓練済みモデルをテストデータセットで検証します。
    
    Args:
        model (YOLO): 訓練済みモデルインスタンス
        data_path (str): データセット設定ファイルのパス
        
    Returns:
        dict: 検証結果の詳細情報
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("モデルの性能検証を開始します...")
        
        # 検証データセットでモデルの性能を評価
        validation_results = model.val(data=data_path)
        
        # 主要な性能指標をログ出力
        if hasattr(validation_results, 'box'):
            box_metrics = validation_results.box
            # mAP (mean Average Precision): 物体検出の主要指標
            logger.info(f"mAP@0.5: {box_metrics.map50:.4f} (IoU閾値0.5での平均精度)")
            logger.info(f"mAP@0.5:0.95: {box_metrics.map:.4f} (IoU閾値0.5-0.95での平均精度)")
            # 精度と再現率: モデルの特性を表す指標
            logger.info(f"Precision: {box_metrics.mp:.4f} (精度: 正しい検出の割合)")
            logger.info(f"Recall: {box_metrics.mr:.4f} (再現率: 実際のオブジェクトを検出できた割合)")
        
        logger.info("モデルの性能検証が完了しました")
        return validation_results
        
    except Exception as e:
        logger.error(f"検証処理が失敗しました: {str(e)}")
        raise


def export_model(model, formats=None, project="weights", name="best_model"):
    """
    訓練済みモデルを各種形式でエクスポートします。
    
    Args:
        model (YOLO): 訓練済みモデルインスタンス
        formats (list): エクスポート形式のリスト
        project (str): エクスポート先ディレクトリ
        name (str): エクスポートファイル名のプレフィックス
    """
    # デフォルトのエクスポート形式を設定（異なるプラットフォームでの使用を想定）
    if formats is None:
        formats = ["onnx", "torchscript"]  # ONNX: 汎用的、TorchScript: PyTorch最適化
    
    logger = logging.getLogger(__name__)
    
    # モデルファイル保存用ディレクトリを作成
    weights_dir = Path(project)
    weights_dir.mkdir(exist_ok=True)
    
    try:
        # 指定された各形式でモデルをエクスポート
        for format_type in formats:
            logger.info(f"モデルを{format_type}形式でエクスポート中...")
            # YOLOv8のエクスポート機能を使用して形式変換
            model.export(format=format_type)
            logger.info(f"{format_type}形式でのモデルエクスポートが成功しました")
    
    except Exception as e:
        # エクスポート処理中のエラーをログに記録
        logger.error(f"モデルエクスポートが失敗しました: {str(e)}")


def main():
    """
    メイン関数：コマンドライン引数を処理してYOLOv8昆虫検出モデルの訓練を実行

    システムチェック、データセット検証、モデル訓練、検証、エクスポートを順次実行します。
    """
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(
        description="YOLOv8を使用した昆虫（カブトムシ）検出モデルの訓練スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python train_yolo.py --data datasets/data.yaml --epochs 100
  python train_yolo.py --data datasets/data.yaml --epochs 50 --batch 16 --device cpu
  python train_yolo.py --data datasets/data.yaml --model yolov8s.pt --export --validate
        """
    )
    
    # 必須コマンドライン引数
    parser.add_argument("--data", type=str, required=True,
                        help="YOLOデータセット設定ファイルのパス (data.yaml)")
    
    # オプションコマンドライン引数
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="使用する事前訓練YOLOv8モデル (デフォルト: yolov8n.pt - Nano版)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="訓練エポック数 - 全データセットを何回繰り返すか (デフォルト: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="訓練時のバッチサイズ - 1回の更新で使用する画像数 (デフォルト: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="訓練用画像リサイズ - 正方形にリサイズされる (デフォルト: 640ピクセル)")
    parser.add_argument("--device", type=str, default="auto",
                        help="計算デバイスの指定 (auto=自動選択, cpu=CPU専用, 0=GPU0, 1=GPU1, 等)")
    parser.add_argument("--project", type=str, default="training_results",
                        help="訓練結果保存用プロジェクトディレクトリ名 (デフォルト: training_results)")
    parser.add_argument("--name", type=str, default="beetle_detection",
                        help="この訓練セッションの実験名 (デフォルト: beetle_detection)")
    parser.add_argument("--export", action="store_true",
                        help="訓練後にモデルをONNX等の他形式でエクスポートする")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="訓練後に検証データセットで性能検証を実行 (デフォルト: True)")
    
    # コマンドライン引数を解析して設定値を取得
    args = parser.parse_args()
    
    # 訓練セッション用のログシステムを初期化
    logger = setup_logging()
    
    # 訓練開始のヘッダーを表示
    logger.info("=" * 60)
    logger.info("YOLOv8 昆虫検出モデル訓練スクリプト")
    logger.info("=" * 60)
    
    # 訓練環境のシステム要件とライブラリバージョンをチェック
    check_system_requirements()
    
    # 指定されたデータセットの構造と内容を検証
    if not validate_dataset(args.data):
        logger.error("データセットの検証が失敗しました。パスやファイル構造を確認してください。")
        sys.exit(1)
    
    try:
        # メインのモデル訓練処理を実行
        model, train_results = train_model(
            data_path=args.data,        # データセット設定ファイル
            model_name=args.model,      # 使用する事前訓練モデル
            epochs=args.epochs,         # 訓練エポック数
            batch_size=args.batch,      # バッチサイズ
            img_size=args.imgsz,        # 画像サイズ
            device=args.device,         # 計算デバイス
            project=args.project,       # プロジェクトディレクトリ
            name=args.name              # 実験名
        )
        
        # オプション: 訓練後のモデル性能検証を実行
        if args.validate:
            validation_results = validate_model(model, args.data)
        
        # オプション: 他のフレームワークで使用できる形式でモデルをエクスポート
        if args.export:
            export_model(model, project="weights", name="beetle_detection_model")
        
        # 訓練パイプラインの成功をユーザーに報告
        logger.info("訓練パイプラインが成功しました")
        logger.info(f"モデル重みの保存先: {args.project}/{args.name}/weights/")
        logger.info("訓練結果を確認して、detect_insect.pyで推論をテストしてください")
        
    except Exception as e:
        # 訓練中の予期しないエラーをログに記録して終了
        logger.error(f"訓練パイプラインが予期しないエラーで失敗しました: {str(e)}")
        logger.error("エラーの原因を確認し、データセットやパラメータを再確認してください")
        sys.exit(1)


# スクリプトが直接実行された場合のみメイン関数を呼び出し
# (モジュールとしてインポートされた場合は実行されない)
if __name__ == "__main__":
    main()