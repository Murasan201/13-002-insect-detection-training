#!/usr/bin/env python3
"""
書籍読者向け統合スクリプト
Book: [Your Book Title] - Chapter X: Insect Detection with YOLOv8

このスクリプトは書籍読者が簡単に昆虫検出を試せるよう設計されています。
ライセンス: MIT (このスクリプト), AGPL-3.0 (使用モデル)
"""

import os
import sys
from pathlib import Path

def print_license_info():
    """ライセンス情報を表示"""
    print("=" * 60)
    print("書籍読者の皆様へ")
    print("=" * 60)
    print("使用モデル: YOLOv8ベース昆虫検出モデル")
    print("ライセンス: AGPL-3.0")
    print("配布元: Hugging Face Model Hub")
    print("")
    print("ライセンス条件:")
    print("  教育・研究利用: 自由")
    print("  個人プロジェクト: 自由")
    print("  商用利用: 要ライセンス確認")
    print("")
    print("詳細: https://www.gnu.org/licenses/agpl-3.0.html")
    print("=" * 60)
    print()

def download_model_if_needed():
    """必要に応じてモデルをダウンロード"""
    weights_dir = Path("./weights")
    model_path = weights_dir / "best.pt"
    
    if model_path.exists():
        print(f"モデルが見つかりました: {model_path}")
        return str(model_path)

    print("昆虫検出モデルをダウンロード中...")
    print("   初回のみ時間がかかります（約6.3MB）")
    
    try:
        from huggingface_hub import hf_hub_download
        
        # weights ディレクトリを作成
        weights_dir.mkdir(exist_ok=True)
        
        # モデルダウンロード
        downloaded_path = hf_hub_download(
            repo_id="murasan/beetle-detection-yolov8",  # 実際のrepo IDに変更
            filename="best.pt",
            local_dir="./weights",
            local_dir_use_symlinks=False
        )
        
        print(f"ダウンロード完了: {downloaded_path}")
        return downloaded_path

    except ImportError:
        print("エラー: huggingface_hub がインストールされていません")
        print("   pip install huggingface_hub を実行してください")
        sys.exit(1)
    except Exception as e:
        print(f"ダウンロードエラー: {e}")
        print("   ネットワーク接続を確認してください")
        sys.exit(1)

def setup_directories():
    """必要なディレクトリを作成"""
    dirs = ["input_images", "output_images", "weights", "logs"]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"ディレクトリ作成: {dir_name}/")

def run_detection_demo():
    """デモ検出を実行"""
    print("\n昆虫検出デモを開始...")

    # input_images にサンプル画像があるかチェック
    input_dir = Path("input_images")
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    if not image_files:
        print("input_images/ にテスト画像がありません")
        print("   サンプル画像を配置してから再実行してください")
        return

    print(f"{len(image_files)} 個の画像を発見")
    
    try:
        # 検出スクリプト実行
        import subprocess
        result = subprocess.run([
            sys.executable, "detect_insect.py",
            "--input", "input_images/",
            "--output", "output_images/"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("検出完了 output_images/ で結果を確認してください")
            print("ログ: logs/ ディレクトリを確認")
        else:
            print(f"検出エラー: {result.stderr}")

    except FileNotFoundError:
        print("エラー: detect_insect.py が見つかりません")
        print("   正しいディレクトリで実行していることを確認してください")

def main():
    """メイン関数 - 書籍読者向けワンクリックセットアップ"""
    print("昆虫検出システム - 書籍版セットアップ")

    # ライセンス情報表示
    print_license_info()

    # ユーザー確認
    response = input("続行しますか？ (y/N): ").lower().strip()
    if response != 'y':
        print("セットアップを中止しました")
        return

    print("\nセットアップ開始...")

    # 1. ディレクトリ作成
    setup_directories()

    # 2. モデルダウンロード
    model_path = download_model_if_needed()

    # 3. デモ実行（オプション）
    if input("\nサンプル検出を実行しますか？ (y/N): ").lower().strip() == 'y':
        run_detection_demo()

    print("\nセットアップ完了")
    print("\n書籍での使用方法:")
    print("   python detect_insect.py --input input_images/ --output output_images/")
    print("\nカスタム検出:")
    print("   1. 画像を input_images/ に配置")
    print("   2. 上記コマンド実行")
    print("   3. output_images/ で結果確認")

if __name__ == "__main__":
    main()