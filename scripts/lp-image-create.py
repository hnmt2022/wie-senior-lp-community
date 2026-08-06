#!/usr/bin/env python3
"""
Wie for Life LP用画像生成スクリプト

Gemini APIを使用し、暮らし・お金・将来の備えを支援する
「Wie for Life」のLP向け画像素材を生成する。

画像の基本方針:
  - Wieのブランドカラー（淡い黄色・紺）
  - シニアを「弱い人」として描かず、自然で前向きな日常を表現
  - 税務・保険・エンディングノートを、過度な終活感なく表現
  - 画像内の文字は原則入れず、HTML/CSSで重ねる

使用例:
    python wie-for-life-lp-image-create.py --preset hero \
        --prompt "明るい自宅で書類を整理しながら穏やかに話す60代女性と女性相談員" \
        --output assets/hero-bg.png

    python wie-for-life-lp-image-create.py --preset consultation \
        --prompt "税金や保険の書類を机に置き、相談員と落ち着いて話す女性" \
        --output assets/consultation.png

    python wie-for-life-lp-image-create.py --preset notebook \
        --prompt "家族に伝えたいことをノートに書き留める手元" \
        --output assets/ending-note.png

    python wie-for-life-lp-image-create.py --preset icon \
        --prompt "書類を一緒に整理することを表す、書類と対話のシンプルなアイコン" \
        --output assets/icon-organize.png

環境変数 — wie-for-life-lp/.env に設定:
    GOOGLE_CLOUD_PROJECT: Vertex AIプロジェクトID
    GOOGLE_CLOUD_LOCATION: リージョン（既定値: global）
    GEMINI_API_KEY: Google Gemini APIキー（Vertex AIを使わない場合）
"""

import argparse
import io
import os
from pathlib import Path
from typing import Optional

from PIL import Image
from google import genai
from google.genai import types


BRAND_DIRECTION = """
【Wie for Lifeのブランド方針】
- 基調色は淡い黄色、アイボリー、白。アクセントに落ち着いた紺色
- やさしさは保ちつつ、幼くせず、専門家に相談できる信頼感を出す
- 明るい自然光、清潔で落ち着いた日本の住空間または相談スペース
- 登場人物は日本で暮らす人として自然に描く
- シニアを一律に白髪、杖、介護状態で表現しない
- 不安、孤独、病気、死を過度に強調しない
- 保険商品の販売、札束、節税の派手な演出は避ける
- 『終わりの準備』ではなく『これからを安心して暮らすための整理』を表現する
"""


LP_PRESETS = {
    "hero": {
        "name": "ファーストビュー",
        "aspect_ratio": "16:9",
        "description": "サービス全体の安心感と、相談しやすさを伝える背景画像",
        "style_hint": """
写実的で自然な写真風の画像にしてください。
- 相談する本人が主役で、支援者が一方的に教える構図にしない
- 人物は中央を避け、見出しを置ける十分な余白を設ける
- 明るく穏やかで、生活の延長として相談できる雰囲気
""",
    },
    "consultation": {
        "name": "相談風景",
        "aspect_ratio": "4:3",
        "description": "税金・保険・暮らしの相談を一緒に整理する様子",
        "style_hint": """
自然な相談風景の写真にしてください。
- 対面で穏やかに話す二人を対等な関係として描く
- 机には少量の書類、ノート、筆記具を置く
- 書類の文字、企業名、保険会社名、金額は判読できないようにする
- 威圧的なオフィスや営業面談の印象を避ける
""",
    },
    "notebook": {
        "name": "エンディングノート作成",
        "aspect_ratio": "4:3",
        "description": "家族への希望や大切な情報を前向きに整理する場面",
        "style_hint": """
ノートを書く手元を中心とした、温かく自然な写真にしてください。
- 葬儀、遺影、墓、暗い部屋など死を直接連想させる要素は入れない
- 家族やこれからの暮らしを考える前向きな時間として表現する
- ノート内の文字は判読できないようにする
""",
    },
    "portrait": {
        "name": "担当者紹介",
        "aspect_ratio": "1:1",
        "description": "専門性と親しみやすさを伝える人物写真",
        "style_hint": """
自然で清潔感のあるポートレート写真にしてください。
- 胸から上、自然な表情、落ち着いた服装
- 背景は白、アイボリー、または明るい相談スペース
- 過剰なビジネス感や医療従事者のような演出を避ける
- 実在の担当者紹介には生成画像を使わず、本人写真を使用する前提の参考素材とする
""",
    },
    "section-bg": {
        "name": "セクション背景",
        "aspect_ratio": "16:9",
        "description": "文章の可読性を損なわない淡い背景素材",
        "style_hint": """
淡い黄色、アイボリー、白、紺を使った控えめな抽象背景にしてください。
- 装飾は少なく、余白を広く取る
- 上に紺色の文章を載せても読みやすい明度にする
- 花柄や高齢者向け広告に見える装飾は避ける
""",
    },
    "icon": {
        "name": "サービスアイコン",
        "aspect_ratio": "1:1",
        "description": "税務、保険、書類整理などを示すシンプルなイラスト",
        "style_hint": """
線を生かしたシンプルなアイコン風イラストにしてください。
- 淡い黄色、紺、白の2〜3色を使用する
- 小さく表示しても意味が分かる単純な形にする
- コイン、札束、盾、ハートなど既視感の強い記号に頼りすぎない
- 同じLP内で並べやすい統一された線幅と余白にする
""",
    },
    "ogp": {
        "name": "OGP画像",
        "aspect_ratio": "16:9",
        "description": "SNSやメッセージで共有した際に表示する画像",
        "style_hint": """
Wie for Lifeの落ち着いた世界観が一目で伝わる画像にしてください。
- 右側または左側にタイトルを重ねられる広い余白を設ける
- 画像内に文字やロゴは生成しない
- 小さな表示でも暗く沈まず、人物やモチーフを詰め込みすぎない
""",
    },
}


LP_STYLES = {
    "wie-warm": "淡い黄色とアイボリーを中心に、自然光のある温かな雰囲気。親しみやすいが甘すぎない。",
    "trustworthy": "白と紺を基調にした端正で信頼感のある雰囲気。堅苦しくせず、清潔で落ち着いた構図。",
    "natural-photo": "広告らしく作り込みすぎない、自然な日本の暮らしを感じる写真。現実的な人物と室内。",
    "minimal": "余白を生かし、要素を絞ったミニマルなデザイン。文章の邪魔をしない。",
    "soft-illustration": "淡い色面と細い紺の線を使った、落ち着いたフラットイラスト。子ども向けにはしない。",
}


def load_env_file() -> None:
    """スクリプト近傍または実行ディレクトリから上方向に.envを探す。"""
    checked = set()
    for start_dir in (Path(__file__).parent, Path.cwd()):
        current = start_dir.resolve()
        while True:
            env_path = current / ".env"
            if env_path not in checked and env_path.is_file():
                print(f".envを読み込み: {env_path}")
                with env_path.open(encoding="utf-8") as env_file:
                    for raw_line in env_file:
                        line = raw_line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            os.environ[key.strip()] = value.strip().strip("'\"")
                return
            checked.add(env_path)
            if current == current.parent:
                break
            current = current.parent


def create_client() -> genai.Client:
    """Vertex AIまたはAPIキー方式でクライアントを作成する。"""
    load_env_file()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if project:
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        print(f"Vertex AIモード: project={project}, location={location}")
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        print("APIキーモード")
        return genai.Client(api_key=api_key)

    raise EnvironmentError(
        "GOOGLE_CLOUD_PROJECTまたはGEMINI_API_KEYを.envに設定してください"
    )


def build_prompt(user_prompt: str, preset: Optional[str], style: Optional[str]) -> str:
    """利用者の指示に、用途・ブランド・共通制約を加える。"""
    parts = []
    if preset:
        selected_preset = LP_PRESETS[preset]
        parts.append(
            f"【用途】{selected_preset['name']} — {selected_preset['description']}"
        )
        parts.append(selected_preset["style_hint"])

    parts.append(BRAND_DIRECTION)

    if style:
        parts.append(f"【スタイル】{LP_STYLES[style]}")

    parts.append("""
【共通の制約】
- Wie for Lifeのランディングページに使用する画像素材
- HTML/CSSで文章を重ねるため、画像内に文字、ロゴ、透かしを入れない
- 税理士、保険、介護などの資格や業務範囲を画像だけで誤認させない
- 実在のお客様の声に見せるための架空の人物写真として使わない
- Web表示に適した、自然で高品質な仕上がりにする
- 指や手、書類、眼鏡などの形を不自然にしない
""")
    parts.append(f"【生成する画像の内容】\n{user_prompt}")
    return "\n".join(parts)


IMAGE_GENERATION_MODELS = [
    "gemini-3-pro-image",
    "gemini-2.5-flash-image",
]


def generate_with_fallback(client, contents, aspect_ratio: str = "16:9"):
    """高品質モデルから順に画像生成を試す。"""
    last_error = None
    for model_name in IMAGE_GENERATION_MODELS:
        try:
            print(f"モデル使用: {model_name}")
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                ),
            )
        except Exception as error:
            print(f"エラー: {model_name} - {type(error).__name__}: {error}")
            last_error = error
    raise RuntimeError(f"すべてのモデルが利用できません。最後のエラー: {last_error}")


def generate_lp_image(
    prompt: str,
    output_path: str,
    preset: Optional[str] = None,
    style: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    reference: Optional[str] = None,
) -> str:
    """Wie for LifeのLP用画像を生成して保存する。"""
    client = create_client()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    selected_ratio = (
        aspect_ratio
        or (LP_PRESETS[preset]["aspect_ratio"] if preset else None)
        or "16:9"
    )

    if reference:
        reference_path = Path(reference)
        if not reference_path.is_file():
            raise FileNotFoundError(f"参考画像が見つかりません: {reference}")
        print(f"画像編集モード: {reference}")
        ref_image = Image.open(reference_path)
        edit_prompt = build_prompt(
            f"参考画像を基に、次の指示で修正してください。\n{prompt}",
            preset,
            style,
        )
        response = generate_with_fallback(client, [edit_prompt, ref_image], selected_ratio)
    else:
        full_prompt = build_prompt(prompt, preset, style)
        print(
            f"プリセット: {preset or 'なし'} / "
            f"スタイル: {style or 'デフォルト'} / "
            f"アスペクト比: {selected_ratio}"
        )
        response = generate_with_fallback(client, full_prompt, selected_ratio)

    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            if part.inline_data is not None:
                image = Image.open(io.BytesIO(part.inline_data.data))
                image.save(output_path)
                print(f"画像を保存しました: {output_path}")
                return output_path

    raise RuntimeError("画像が生成されませんでした。プロンプトを変えて再試行してください")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wie for LifeのLP用画像素材をGeminiで生成します"
    )
    parser.add_argument("--prompt", "-p", help="生成する画像の内容")
    parser.add_argument("--output", "-o", help="出力ファイルパス")
    parser.add_argument(
        "--preset",
        choices=list(LP_PRESETS),
        help="用途プリセット",
    )
    parser.add_argument(
        "--style",
        "-s",
        choices=list(LP_STYLES),
        default="wie-warm",
        help="画像スタイル（既定値: wie-warm）",
    )
    parser.add_argument(
        "--aspect-ratio",
        "-a",
        choices=["1:1", "16:9", "9:16", "4:3", "3:4"],
        help="アスペクト比。未指定時はプリセットから決定",
    )
    parser.add_argument("--reference", "-r", help="編集元となる参考画像")
    parser.add_argument("--list-presets", action="store_true", help="一覧を表示")
    args = parser.parse_args()

    if args.list_presets:
        print("\n=== Wie for Life LP用プリセット ===\n")
        for key, preset in LP_PRESETS.items():
            print(
                f"  {key:14} {preset['aspect_ratio']:5}  "
                f"{preset['name']} — {preset['description']}"
            )
        print("\n=== スタイル ===\n")
        for key, description in LP_STYLES.items():
            print(f"  {key:18} {description}")
        return

    if not args.prompt or not args.output:
        parser.error("--promptと--outputは必須です（一覧表示時を除く）")

    generate_lp_image(
        prompt=args.prompt,
        output_path=args.output,
        preset=args.preset,
        style=args.style,
        aspect_ratio=args.aspect_ratio,
        reference=args.reference,
    )


if __name__ == "__main__":
    main()
