#!/usr/bin/env python3
"""
カブトムシ検出データセット セットアップスクリプト
手動ダウンロードしたZIPファイルを展開し、YOLOv8訓練用のディレクトリ構成に配置する
要件定義書: docs/insect_detection_application_test_project_requirements_spec.md
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


# ===== 定数定義 =====
# デフォルトのディレクトリパス
DOWNLOADS_DIR = "downloads"
DATASETS_DIR = "datasets"

# データセット情報（ライセンス表示用）
DATASET_URL = "https://universe.roboflow.com/z-algae-bilby/beetle/dataset/1"
DATASET_LICENSE = "CC BY 4.0"
DATASET_AUTHOR = "z-algae-bilby"


def find_zip_files(downloads_dir):
    """
    ダウンロードディレクトリ内のZIPファイルを検索する

    Args:
        downloads_dir (str): ZIPファイルを探すディレクトリパス
    """
    downloads_path = Path(downloads_dir)

    # ディレクトリが存在しない場合は空リストを返す
    if not downloads_path.exists():
        return []

    # .zip拡張子のファイルをすべて取得
    zip_files = list(downloads_path.glob("*.zip"))
    return zip_files


def extract_dataset(zip_path, output_dir):
    """
    ZIPファイルを指定ディレクトリに展開する

    Args:
        zip_path (Path): 展開するZIPファイルのパス
        output_dir (str): 展開先ディレクトリ（存在しない場合は自動作成）
    """
    output_path = Path(output_dir)

    print(f"ZIPファイルを展開中: {zip_path.name}")
    print(f"   展開先: {output_path}")
    print()

    try:
        # 展開先ディレクトリを作成（親ディレクトリも含めて作成）
        output_path.mkdir(parents=True, exist_ok=True)

        # ZIPファイルを開いて展開
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 展開前にファイル数を表示
            file_list = zip_ref.namelist()
            print(f"   {len(file_list)} 個のファイルを展開します...")

            # 全ファイルを展開
            zip_ref.extractall(output_path)

        print("展開完了")
        return True

    except zipfile.BadZipFile:
        # ZIPファイルが破損している場合
        print("展開エラー: ZIPファイルが破損しています")
        print("ヒント: ファイルを再ダウンロードしてください")
        return False

    except PermissionError as e:
        # 書き込み権限がない場合
        print(f"展開エラー: 書き込み権限がありません - {e}")
        print("ヒント: 出力ディレクトリの権限を確認してください")
        return False

    except Exception as e:
        print(f"展開エラー: {e}")
        return False


def verify_dataset_structure(dataset_dir):
    """
    展開したデータセットがYOLOv8形式の構造になっているか検証する

    Args:
        dataset_dir (str): 検証するデータセットのディレクトリ
    """
    dataset_path = Path(dataset_dir)

    print()
    print("データセット構造を検証中...")
    print()

    # YOLOv8形式で必須のディレクトリ構造
    # 形式: "相対パス": "説明（日本語）"
    required_structure = {
        "train/images": "訓練用画像",
        "train/labels": "訓練用ラベル（YOLO形式のtxtファイル）",
        "valid/images": "検証用画像",
        "valid/labels": "検証用ラベル",
    }

    # data.yamlの存在確認（YOLOv8の設定ファイル）
    data_yaml = dataset_path / "data.yaml"
    if data_yaml.exists():
        print("data.yaml: 見つかりました")
    else:
        print("data.yaml: 見つかりません")
        # サブディレクトリにdata.yamlがある場合はルートにコピー
        for yaml_file in dataset_path.rglob("data.yaml"):
            print(f"   -> {yaml_file} に見つかりました")
            if yaml_file.parent != dataset_path:
                shutil.copy(yaml_file, data_yaml)
                print("   -> ルートディレクトリにコピーしました")
            break

    # 必須ディレクトリの存在確認とファイル数カウント
    all_found = True
    stats = {}

    for dir_path, description in required_structure.items():
        full_path = dataset_path / dir_path
        if full_path.exists():
            # ディレクトリ内のファイル数をカウント（サブディレクトリは除外）
            files = list(full_path.glob("*"))
            file_count = len([f for f in files if f.is_file()])
            stats[dir_path] = file_count
            print(f"{dir_path}: {file_count} ファイル ({description})")
        else:
            print(f"{dir_path}: 見つかりません ({description})")
            all_found = False

    # testディレクトリの確認（オプション：なくても訓練は可能）
    test_images = dataset_path / "test/images"
    test_labels = dataset_path / "test/labels"
    if test_images.exists() and test_labels.exists():
        test_count = len([f for f in test_images.glob("*") if f.is_file()])
        print(f"test/images: {test_count} ファイル (テスト用画像・オプション)")

    print()

    if all_found:
        print("データセット構造の検証: 成功")

        # 統計サマリーを表示
        total_train = stats.get("train/images", 0)
        total_valid = stats.get("valid/images", 0)
        print()
        print("データセット統計:")
        print(f"   訓練画像: {total_train} 枚")
        print(f"   検証画像: {total_valid} 枚")
        print(f"   合計: {total_train + total_valid} 枚")

        return True
    else:
        print("データセット構造の検証: 一部のディレクトリが見つかりません")
        print("ヒント: YOLOv8形式でエクスポートしたデータセットを使用してください")
        return False


def print_download_instructions():
    """
    ZIPファイルが見つからない場合のダウンロード手順を表示する
    """
    print()
    print("=" * 60)
    print("データセットのダウンロード手順")
    print("=" * 60)
    print()
    print("1. ブラウザで以下のURLにアクセス:")
    print(f"   {DATASET_URL}")
    print()
    print("2. 「Download Dataset」ボタンをクリック")
    print()
    print("3. フォーマットで「YOLOv8」を選択")
    print()
    print("4. ダウンロードしたZIPファイルを以下に配置:")
    print(f"   {DOWNLOADS_DIR}/")
    print()
    print("5. このスクリプトを再実行:")
    print("   python3 setup_dataset.py")
    print()
    print("=" * 60)


def main():
    """
    メイン関数：ZIPファイルを検索し、展開・検証を実行する
    """
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(
        description="手動ダウンロードしたデータセットZIPを展開・配置する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
使用例:
  python3 setup_dataset.py
  python3 setup_dataset.py --downloads my_downloads --output my_datasets
  python3 setup_dataset.py --delete-zip  # 展開後にZIPを削除

データセット情報:
  提供元: Roboflow Universe ({DATASET_AUTHOR})
  ライセンス: {DATASET_LICENSE}
  URL: {DATASET_URL}
        """
    )

    # ZIPファイルの配置ディレクトリ
    parser.add_argument(
        "--downloads", "-d",
        type=str,
        default=DOWNLOADS_DIR,
        help=f"ZIPファイルが配置されているディレクトリ (デフォルト: {DOWNLOADS_DIR})"
    )

    # 展開先ディレクトリ
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=DATASETS_DIR,
        help=f"データセットの展開先ディレクトリ (デフォルト: {DATASETS_DIR})"
    )

    # ZIP削除オプション（デフォルトは保持）
    parser.add_argument(
        "--delete-zip",
        action="store_true",
        help="展開後にZIPファイルを削除する（デフォルト: 保持）"
    )

    args = parser.parse_args()

    # ヘッダー表示
    print()
    print("=" * 60)
    print("カブトムシ検出データセット セットアップ")
    print("=" * 60)
    print()

    # データセットのライセンス情報を表示（CC BY 4.0の帰属表示要件）
    print("データセット情報:")
    print(f"   提供元: Roboflow Universe ({DATASET_AUTHOR})")
    print(f"   ライセンス: {DATASET_LICENSE}")
    print(f"   URL: {DATASET_URL}")
    print()

    # ダウンロードディレクトリの確認・作成
    downloads_path = Path(args.downloads)
    if not downloads_path.exists():
        print(f"ダウンロードディレクトリを作成: {downloads_path}")
        downloads_path.mkdir(parents=True, exist_ok=True)

    # ZIPファイルを検索
    zip_files = find_zip_files(args.downloads)

    # ZIPファイルが見つからない場合はダウンロード手順を表示して終了
    if not zip_files:
        print(f"{args.downloads}/ ディレクトリにZIPファイルが見つかりません")
        print_download_instructions()
        sys.exit(1)

    # ZIPファイルの選択（複数ある場合はユーザーに選択させる）
    if len(zip_files) == 1:
        selected_zip = zip_files[0]
        print(f"ZIPファイルを発見: {selected_zip.name}")
    else:
        # 複数のZIPファイルがある場合は一覧表示
        print("複数のZIPファイルが見つかりました:")
        for i, zip_file in enumerate(zip_files, 1):
            size_mb = zip_file.stat().st_size / (1024 * 1024)
            print(f"   {i}. {zip_file.name} ({size_mb:.1f} MB)")

        # ユーザーに選択を求める
        print()
        while True:
            try:
                choice = input(f"使用するファイルを選択 (1-{len(zip_files)}): ")
                idx = int(choice) - 1
                if 0 <= idx < len(zip_files):
                    selected_zip = zip_files[idx]
                    break
                print("無効な選択です。範囲内の数字を入力してください。")
            except ValueError:
                print("数字を入力してください。")
            except KeyboardInterrupt:
                print("\nキャンセルしました")
                sys.exit(0)

    print()

    # ZIPファイルを展開
    if not extract_dataset(selected_zip, args.output):
        print()
        print("データセットの展開に失敗しました")
        sys.exit(1)

    # データセット構造を検証
    if not verify_dataset_structure(args.output):
        print()
        print("データセット構造に問題がある可能性があります")
        print("手動で確認してください")

    # ZIPファイルの処理（デフォルト: 保持）
    if args.delete_zip:
        print()
        print(f"ZIPファイルを削除: {selected_zip.name}")
        selected_zip.unlink()
    else:
        print()
        print(f"ZIPファイルを保持: {selected_zip}")

    print()
    print("データセットの準備が完了しました")


if __name__ == "__main__":
    main()
