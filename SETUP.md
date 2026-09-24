# セットアップ（まっさらな Mac から）

> 導入・設定は**自己責任**でお願いします。無料のサポート窓口はありません
> （[README のサポートについて](README.md#サポートについて)）。うまくいかないところは、
> コードを読んで直せる前提で公開しています。有償での対応が必要な方はご相談ください。

所要 20〜30 分（モデルの取得待ちが大半）。**最初は手元だけ・無料で動きます。**
外のモデル（Deepgram・Gemini）を使うかは、動かしてから決めてください。

---

## 0. 必要なもの

| | |
|---|---|
| **macOS 12.3 以上** | 相手の声を ScreenCaptureKit で拾うため。Windows / Linux では動きません |
| **Apple Silicon**（M1 以降） | 手元で文字起こしする場合。Intel Mac では外（Deepgram・Gemini）を使ってください |
| 空き容量 5 GB 程度 | 文字起こしのモデル 1.5 GB ＋ 要約のモデル 3〜10 GB（手元で回す場合） |
| マイク | ヘッドセット推奨。**スピーカーで聞くと自分のマイクに回り込みます** |

## 1. 下ごしらえ（Homebrew）

買ったままの Mac には、どれも入っていません。上から順にどうぞ。

```bash
# ① Homebrew（https://brew.sh のとおり）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/Install/HEAD/install.sh)"
```

**Apple Silicon の Mac では、入れたあとに PATH を通す必要があります。**
インストーラが最後に「Next steps」として同じことを表示するので、そのとおりに:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

これを飛ばすと、次の `brew` が **command not found** になります。

```bash
# ② この道具が使うもの
brew install python@3.13 ffmpeg
```

- `ffmpeg` は**会議のあとの作り直し**（録音から全文を起こし直す）と録音の圧縮に使います。
  無いと、会議は録れても**そのあとが全部できません**
- **C コンパイラ**（Xcode Command Line Tools）も要ります。依存のうち `webrtcvad` だけ
  出来合いのものが無く、入れるときにその場で組み立てるためです。Homebrew を入れると
  一緒に入りますが、入っていなければ `xcode-select --install`

## 2. 取得と Python の環境

```bash
git clone <このリポジトリの URL> meeting-copilot
cd meeting-copilot

python3.13 -m venv .venv          # `python3` ではなく、入れた 3.13 を名指しする
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**`python3` と書かないでください。** macOS に最初から入っている `/usr/bin/python3` は
**3.9** で、画面音声の取り込みに使う `pyobjc-framework-ScreenCaptureKit`（3.10 以上）が入らず、
`No matching distribution found` で止まります。Homebrew で入れた `python3.13` を名指しするのが確実です
（`python3 -V` が 3.13 以上ならそのままでも構いません）。

`requirements.txt` は `setuptools<81` を指定しています（81 で `pkg_resources` が消え、
声紋の依存が動かなくなるため）。外さないでください。

**入ったか、足りないものは何かを 1 枚で見る:**

```bash
python scripts/check_prereqs.py
```

この Mac の条件（macOS の版・Apple Silicon・空き容量）、依存が入ったか、`ffmpeg` や
`ollama` があるか、画面収録の許可があるかまで、**必須と任意を分けて**出します。
`pip install` の前でも動くので、詰まったらまずこれを見てください。

テストまで動かすなら:

```bash
pip install pytest        # テストを動かすときだけ必要（本体の動作には要りません）
python -m pytest tests -q # 900 件ほど。数十秒で終わります
```

## 3. macOS の許可（ここを飛ばすと「録れているつもりで無音」になります）

1. **画面収録**: システム設定 → プライバシーとセキュリティ → **画面収録** で、
   **ターミナル（または iTerm など、起動に使うアプリ）** を許可 → **アプリを再起動**
   - 相手の声はシステム音声から拾うので、これが無いと**相手側が完全に無音**になります
   - **`会議アシスタント.app`（6 章）から使うときは、許可先が「Python」になります**（ターミナルではありません）。
     許可が無いまま会議を始めると、一覧に Python が載り、案内に Python.app の場所が出ます。
     一覧に無いときは「＋」を押し、`Cmd+Shift+G` でその場所を貼り付けて選びます
     （Homebrew なら `/opt/homebrew/opt/python@3.x/Frameworks/Python.framework/Versions/3.x/Resources/Python.app`）
   - Python の版を上げたとき、macOS の更新のあと、定期的に出る「引き続き許可しますか」を閉じたときは、
     許可が外れることがあります。一覧でオンに見えていても外れていることがあるので、オフ→オンで入れ直します
2. **マイク**: 同じ画面の **マイク** で、同じアプリを許可

**「ネットワークへのアクセスを許可しますか」と聞かれたら、「許可しない」で構いません。**
聞いているのは macOS のファイアウォールで、**外から入ってくる接続を受けるか**だけを見ています。
この道具の画面はどれも `127.0.0.1`（自分の中）にしか口を開かないので、許可しなくても開きます。
Homebrew の Python は版が上がるたびに別のアプリ扱いになるため、何度も聞かれることがあります
（煩わしければ、システム設定 →「ネットワーク」→「ファイアウォール」→「オプション」で
一度だけ許可してください。持ち主が決める設定なので、この道具からは触りません）。

確かめ方:

```bash
python scripts/sck_probe.py          # システム音声が取れているか（何か音を鳴らしながら）
python scripts/check_audio_routing.py # 入出力の経路を見る
```

## 4. 設定

```bash
cp config/settings.example.yaml config/settings.yaml
```

`config/settings.yaml` で、最低限この 2 つを自分のものに:

```yaml
meeting:
  self_name: 自分            # 自分の発言に付く名前
  mic_candidates:            # 番号ではなく名前で書く（機器の番号は挿すたびに変わります）
    - AirPods Pro
    - MacBook Proのマイク
```

マイクの名前が分からなければ、一度起動すると候補の一覧が出ます。

## 5. 手元だけで動かす（無料・既定）

文字起こしと要約を手元で回します。**ネットが切れていても動きます。**

```bash
# 要約に使うローカル LLM（Ollama）
brew install ollama
ollama serve &
ollama pull gemma4          # 設定の llm.model と同じ名前にする
```

```yaml
# config/settings.yaml
stt:
  engine: local             # 手元の mlx-whisper
llm:
  engine: local             # 手元の Ollama
```

初回の起動時に、文字起こしのモデル（約 1.5 GB）が自動で取得されます。

## 6. 起動する

```bash
source .venv/bin/activate
python -m src.main --mode meeting_loopback
```

ブラウザで画面（`http://127.0.0.1:8765/`）が開きます。**以降の操作はすべて画面です。**

ターミナルを開かずに始めたいときは、**リポジトリ直下の `会議アシスタント.command` をダブルクリック**し、
右上の「会議を始める」を押してください（初回だけ Finder で右クリック →「開く」）。
直下に置いてある入口は**この 1 つだけ**です。会議そのものを起こす役は `scripts/会議を始める.command` で、
上のボタンがターミナルで開きます（急ぐときは直接ダブルクリックしても構いません）。

**メニューバーに常駐させる（任意）**: `scripts/アプリを作る.command` をダブルクリックすると
`/Applications/会議アシスタント.app` ができます。メニューバーに 🎙 が出て、記録している間は 🔴 に
変わります。

| メニュー | 何が起きるか |
|---|---|
| （1 行目） | 待機中／記録中の会議名と経過時間 |
| 会議を始める… | `scripts/会議を始める.command` をターミナルで開く |
| 会議の画面を開く | 記録中のダッシュボードをブラウザで開く |
| 会議アシスタントを開く | 開いていなければ立ち上げる |
| ログインしたら起動する | 自動起動の出し入れ |

- **会議そのものには触りません**（開いているポートを覗いて、記録中かどうかを見るだけ）
- **`.app` は手元で作ります。** 署名していない `.app` を配ると、ダウンロードした Mac では
  Gatekeeper に隔離されて開けないためです。Python やリポジトリの置き場所を変えたら、
  もう一度ダブルクリックして作り直してください
- **メニューバーが埋まっていると出ないことがあります**（ノッチのある Mac で実測）。
  API 上は表示されている扱いなのに描画されません。ほかの常駐アプリを 1 つ減らすか、
  メニューバーの整理ツールで隠すと出ます

1. **Zoom に入る前に起動**する
2. 画面の**セルフチェック**でマイクを測る（「測る（3秒）」を押して話す）
3. Zoom に参加する
4. 相手が話している状態で、相手側も「測る（3秒）」
5. 「このまま開始」で本編へ
6. 終わったら画面の「会議を終了」

会議のあと、録音から全文を作り直して、`workspace/sessions/<会議>/minutes.md` に**議事録**が出ます。
作る手段が 1 つも無い環境では、代わりに `minutes_prompt.md` ができます（中身をまるごと AI チャットに貼る）。

### 6-1. 議事録と裏取り（🔍）の手段

手元にあるものを見て、上から自動で選びます（[README の表](README.md)）。

| | 1 番目 | 2 番目 | 3 番目 | 最後 |
|---|---|---|---|---|
| 議事録 | Claude CLI（サブスク） | Gemini（その会議で承認したとき） | Ollama | 貼り付け用の指示書 |
| 裏取り | Claude CLI（サブスク） | Gemini＋Google 検索（同上） | — | 使えない（押すと理由が出る） |

固定したいときだけ設定します:

```yaml
# config/settings.yaml
dispatch:
  on_finish: minutes        # minutes（議事録まで作る）| none（材料だけ書き出して終わる）
minutes:
  engine: auto              # auto | claude_cli | gemini | local | none
meeting:
  verify_engine: auto       # auto | claude_cli | gemini | none
```

会議のあとで作り直したいとき（手段を変えて試す、など）:

```bash
python scripts/make_minutes.py workspace/sessions/<会議>                  # 自動で選ぶ
python scripts/make_minutes.py workspace/sessions/<会議> --engine local   # 手元の LLM で
python scripts/make_minutes.py workspace/sessions/<会議> --engine none    # 貼り付け用の指示書だけ
```

Claude CLI を使うには、[Claude Code](https://docs.anthropic.com/en/docs/claude-code) を入れて
`claude` でログインしておきます（サブスクの範囲で動き、従量課金は増えません）。

### 6-1-1. 会議が終わったら（仕上げは押してから）

会議の終わりに「**ここまでは残りました**」（録音・会議中の全文・議事録の材料・決定事項）と出て、いったん止まります。
そのあとの**録音からの作り直し・声の台帳・辞書の候補・議事録**は、画面の「**いま仕上げる**」を押したときだけ走ります。

- **答えないまま 10 分たつと「あとでやる」に倒れます**（勝手に数分の処理を始めません）
- 対面の商談ですぐ移動する・Wi-Fi が不安定・次の予定が詰まっている、という場面で
  「処理の途中で落ちて積む」を避けるためです
- **あとでやる**を選んだら、`会議アシスタント.command` →「会議のあと」から続きを走らせます
  （ブラウザを閉じても、別の日でも再開できます。端末なら `python scripts/finish_meeting.py workspace/sessions/<会議>`）
- いつも自動で仕上げたいときは `meeting.finish_mode: auto`（`ask` が既定・`later` は常に後回し）

### 6-2. 会議アシスタント（辞書と声の台帳を画面で直す）

置き換え辞書と声の台帳は、**画面「会議アシスタント」で直します**（ファイルを開いて編集しなくていい）。

- **会議中**: ダッシュボードの「辞書に足す」に、誤変換と正しい表記を入れて「足す」（**次の行から効きます**）。
  字幕の文字をマウスで選ぶと、誤変換の欄に入ります。一覧を見る・外すときは「会議アシスタントを開く」
- **会議のあと**: 辞書の候補があれば、この画面が**自動で開きます**。入れたいものにチェックを付けて「選んだものを辞書に入れる」
- **それ以外のとき**: `会議アシスタント.command` をダブルクリック（60 分触らなければ自分で閉じます）
  この画面からは**会議も始められます**（「会議を始める」ボタン）。過去の会議を振り返ってから始められます

| タブ | できること |
|---|---|
| 会議 | 終わった会議の一覧。名前を押すと、会議中と同じ画面（文字起こし・録音の再生・決定事項・TODO）が開く。仕上げが残っていれば「仕上げる」 |
| 辞書の候補 | 会議ごとの候補を選んで入れる／見送る。正しい表記はその場で直してから入れられる |
| 辞書 | 置き換えを足す・外す、守り札（壊さない語）を足す・外す。外すときは直す前の辞書を退避します |
| 声の台帳 | 覚えた人の名前を直す・台帳から外す・声を聞いて確かめる |
| 設定 | **動かし方（下の 6-5）と、機能の ON/OFF**。設定ファイルを開かずに切り替えられる |

**会議の詳細でできること**（会議の名前を押したあと）:

- 行の**時刻を押すとそこから録音が再生**される。✎ で文字と話者を直す
- **参加者の「名前を付ける」** … 「不明話者」に名前を付けると、その会議の全文が書き換わり、
  声の台帳にも覚える。1 人の「不明話者」に**複数人が混ざっている**ときは、
  行の 👤 で **ここから／ここまで**の範囲を選んで付ける
- 議事録・全文・録音の**ダウンロード**

**候補の出どころ**（会議が終わると自動で作ります）:

- **議事録を作った LLM が挙げた誤変換**（議事録と同じ 1 回の呼び出しで取るので、追加の費用や時間はほぼ無い）
- **参加者の名前や事前資料の「## 用語」に近いが違う表記**（手元だけ・LLM なし）
- **辞書の置き換えが別の語を壊した跡**（例: 社名「MDクラウド」が「クラウド→Claude」で「MDClaude」になった → 守り札の候補）

**自動では辞書に入れません。** 置き換えは全行に効くので、1 語の誤りがほかの語を壊します。
辞書に向くのは何度も同じ形で出る語（社名・製品名・人名）で、1 回だけの言い間違いは見送ってください。
入れる先は、その会議の文字起こしに使ったエンジンの辞書です（手元の Whisper と外の Gemini で誤り方が違うため、辞書も別）。
事前資料に `## 用語` の箇条書きで社名・製品名を書いておくと、その会議のあいだは**辞書がその語を壊さなくなり**、
要約と議事録もその表記にそろいます（辞書に足さなくても効きます）。

### 6-3. 前にも出た人を候補に出す（声の台帳・任意）

会議のあとに、**名前を付けた人の声**を手元の台帳に覚え、次の会議でその人が「不明話者」として出てきたら、
参加者の欄に **「〇〇さん？」** と候補を出します。押せば名前が付き、過去の行にも反映されます。

```yaml
meeting:
  voice_library:
    enabled: true           # 既定は false。声紋は個人を識別できる情報です（SECURITY.md の 4）
```

**自動では名前を付けません。** 会議をまたぐと、マイクや回線の違いで本人と別人の声の近さが重なるためです
（実測: 別の会議の本人 0.85〜0.92／別人 〜0.85）。押した／断ったは会議ごとに記録されます。

### 6-4. 画面共有を控える（何のページの話だったかを残す・任意）

相手が画面を共有しているとき、サイト名や URL は**読み上げられません**。あとでそのページを探すのが
大変なので、**会議アプリのウィンドウだけ**を一定間隔で撮り、**変わったときだけ**残します。
文字は手元（macOS の Vision）で読み、**URL とページ名**を控えます。会議の詳細画面では、
その時刻の発言の隣にキャプチャが出ます。

```yaml
meeting:
  screen_capture:
    enabled: true           # 既定は false
    owners: [zoom.us, Zoom, Google Chrome, Comet]   # この 4 つ以外は撮らない
    interval_sec: 6         # 何秒ごとに見るか
    change_threshold: 0.02  # 前の 1 枚と違う画素がこの割合を超えたら残す
    max_shots: 400          # 1 会議で残す上限
```

- **画面全体は撮りません**（指定したアプリのウィンドウだけ）。許可は 3. の**画面収録**と同じものです
- **外へは何も送りません**（撮るのも読むのも手元で完結）
- 残るのは `workspace/sessions/<会議>/screens/<秒>.jpg` と `screens.jsonl`（時刻・URL・ページ名）
- **相手の資料そのものが手元に残ります。** 使うかどうかは運用として決めてください（[SECURITY.md](SECURITY.md) の 4）
- 文字を読まずに画像だけ残すなら `ocr: false`

### 6-5. 動かし方を選ぶ（プリセット）

処理ごとに「どこで回すか」を選べます。1 つずつ触らなくても、**設定タブの「動かし方」で
まとめて決められます**。

| 動かし方 | 文字起こし | 会議中の要約 | 議事録 | 裏取り | 費用の目安（月 15 時間） |
|---|---|---|---|---|---|
| 機密優先 | 手元の Whisper | 手元の Ollama | 手元の LLM | なし | ¥0 |
| コスト優先 | 手元の Whisper | 手元の Ollama | Claude CLI | Claude CLI | ¥0（サブスクの範囲） |
| バランス | Deepgram | 手元の Ollama | Claude CLI | Claude CLI | 月 ¥250 ほど |
| 速さ優先 | Deepgram | Gemini | Claude CLI | Claude CLI | 月 ¥900 ほど |

- **この Mac でできない動かし方は押せません**（理由が画面に出ます）。判定に見るのは
  Apple Silicon か／Ollama とモデルがあるか／Claude CLI があるか／Gemini のキーがあるか／メモリ。
  **会議が始まってから「動きません」と分かるのを避ける**ためです
- 「手元で回せば軽くなる」は**逆**です。外に出さない代わりに自分の GPU を使います。
  実測（36GB のマシン）では、手元だけで回すと会議中の文字起こしが 1〜2 分遅れました。
  メモリ 24GB 未満のマシンでは、オフラインの選択肢は出ません
- 費用は開発元の実測（Deepgram ¥39/時間・Gemini ¥83/時間）です。会議の長さと話す量で変わります
- **会議中の要約を作り直す間隔**（既定 2 分）も設定タブで変えられます（30 / 60 / 120 / 180 秒）。
  実測では 30 秒にすると 58% が空振り（前回から何も変わらない）で、送る量だけが増えました
- 端末の能力だけ見たいときは `python -m src.capability`

## 7. 鍵（API キー）の早見表 — 何が要るか、どこで取るか、どこに置くか

**手元だけで使うなら、鍵はひとつも要りません。** 下は「外のエンジンも使う」場合の話です。

| 使いたいもの | 何を登録するか | 取る場所 | 置く場所 | 費用 |
|---|---|---|---|---|
| **会議中の文字起こしを速くする**（推奨） | Deepgram のアカウント | [console.deepgram.com](https://console.deepgram.com/) → **API Keys** | `~/.config/meeting-copilot/deepgram.key` | 約 ¥39/時間 |
| **文字起こし・要約を Gemini で** | Google Cloud のプロジェクト（**課金必須**）＋ API キー | [Google AI Studio](https://aistudio.google.com/) → **Get API key** | `~/.config/meeting-copilot/gemini.key` | 約 ¥83/時間 |
| **議事録と裏取りを賢くする** | Claude のサブスク（Pro / Max） | [Claude Code](https://docs.anthropic.com/en/docs/claude-code) を入れて `claude` でログイン | **鍵は要りません**（ログイン状態を使う） | サブスクの範囲内 |
| **要約を手元で回す** | — | — | **鍵は要りません**（Ollama） | 無料 |
| **議事録を自分の道具へ送る**（Notion・Slack など） | つなぎ先それぞれのトークン | 各サービスの管理画面 | **自分で書く**（10 章） | — |

**鍵は設定ファイルにもリポジトリにも書きません。** この道具が持つのは**ファイルの場所**だけです
（`deepgram_key_file` / `key_file`）。どのファイルも `chmod 600`（自分だけが読める）にしてください。

**置き場所を変えたいとき**は、設定のパスを書き換えれば、どこでも構いません。

### 7-0. 鍵が効いているかを確かめる

```bash
python scripts/check_gemini_key.py      # Gemini: 課金の確認 → モデル一覧が引けるか（音声は送りません）
```

Deepgram は、会議のはじめに **0.3 秒の無音で 1 回試す**ので、鍵が違えば
その場で手元の文字起こしに倒れます（画面に理由が出ます）。

## 8. 外のエンジンを使う場合だけ（Deepgram / Gemini）

手元で足りていれば、ここは飛ばして構いません。外へ出すと、**画面に出るまでが 17 分 → 約 9.5 秒**
（Deepgram）になり、取りこぼしも減ります（実測: 6.4% → 0.5%）。

**どちらを使うか**（開発元の実測・文字の食い違い率で精度に差はありません）:

| | 画面に出るまで | 費用 | 送るもの |
|---|---|---|---|
| **Deepgram**（文字起こし） | 約 9.5 秒 | ¥39/時間 | 音声 |
| **Gemini**（文字起こし） | 約 43 秒 | ¥83/時間 | 音声 |
| **Gemini**（会議中の要約・議事録・裏取り） | — | ¥83/時間の内訳の 57% | テキストだけ |

会議中の文字起こしは Deepgram、要約や議事録の賢さが要るところは Gemini、という分け方ができます。
自分の音源で測り直すなら `python scripts/bench_asr.py --audio <音声> --reference <正解のテキスト>`。

### 8-1. Deepgram を使う（文字起こしだけを外へ）

1. [deepgram.com](https://deepgram.com/) でアカウントを作る（メールアドレスか Google / GitHub）
2. [console.deepgram.com](https://console.deepgram.com/) を開き、左の **API Keys** →
   **Create a New API Key**。権限は読み書きの既定のままで構いません
3. **表示されるのはその 1 回だけ**です。閉じる前にコピーする
4. 下のとおりファイルに置く（`ここにキー` の所に貼る）

```bash
mkdir -p ~/.config/meeting-copilot
printf '%s' 'ここにキー' > ~/.config/meeting-copilot/deepgram.key
chmod 600 ~/.config/meeting-copilot/deepgram.key
```

5. 設定（下）を入れて、会議を 1 本試す。**画面の右下に「文字起こし: Deepgram」と出れば効いています**
   （鍵が読めない・通らないときは、黙って手元の Whisper に倒れ、理由が画面に出ます）

```yaml
stt:
  engine: deepgram
meeting:
  external_stt:
    enabled: true
    provider: deepgram
    deepgram_key_file: ~/.config/meeting-copilot/deepgram.key
    deepgram_model: nova-3
```

- 送る要求には毎回「モデル改善に使わない」印（`mip_opt_out`）が付きます。**設定では外せません。**
  この印が付いた要求は、応答を返したあと Deepgram 側に**保存もされません**
- 会議のはじめに **その形の要求を受け付けるかを 0.3 秒の無音で 1 回試し**、通らなければ手元に倒れます
  （プロジェクト設定の opt out は、更新しても黙って無視されることがあるため、毎回の要求で守ります）
- 話者分けは使いません（**手元の声紋のほうが当たります**。実測 97% 対 76%）

### 8-2. Gemini を使う: 有料枠の Google Cloud プロジェクトを用意する

**無料枠は使えません**（送った内容が製品改善に使われ、人間が読むことがあります。規約に明記）。
この道具は**課金が有効か機械的に確かめてから**でないと音声を送りません。

```bash
# Google Cloud SDK を入れて、ログインする
brew install --cask google-cloud-sdk
gcloud auth login

# 会議専用のプロジェクトを作り、課金を紐づける（ほかの用途と混ぜない＝請求を分けるため）
gcloud projects create my-meeting-stt
gcloud billing projects link my-meeting-stt --billing-account <請求先アカウント ID>
gcloud services enable generativelanguage.googleapis.com --project my-meeting-stt
```

### 8-3. API キーを置く

[Google AI Studio](https://aistudio.google.com/) で、**そのプロジェクトの**キーを作ります。
キーは `generativelanguage.googleapis.com` だけに制限してください。

```bash
mkdir -p ~/.config/meeting-copilot
printf '%s' 'ここにキー' > ~/.config/meeting-copilot/gemini.key
chmod 600 ~/.config/meeting-copilot/gemini.key
```

**キーをリポジトリの中や設定ファイルに書かないでください。** この道具はファイルの場所だけを
設定に持ちます（`meeting.external_stt.key_file`）。

### 8-4. 設定を入れる

```yaml
stt:
  engine: gemini
llm:
  engine: gemini
meeting:
  external_stt:
    enabled: true
    billing_project: my-meeting-stt
    key_file: ~/.config/meeting-copilot/gemini.key
```

### 8-5. 送らずに刻みだけ見る → 実際に送る

```bash
python scripts/make_test_session.py --name selftest      # 合成音声（macOS の say）で試験用の会議を作る
python scripts/selftest_live.py                          # 送らない。刻み方と回数だけ見る
python scripts/selftest_live.py --send --limit 2         # 実際に 2 窓だけ送る（数円）
```

**本物の会議の録音で試さないでください。** 実在の相手の声は試験に使えません。

## 9. よくあるつまずき

| 症状 | 見るところ |
|---|---|
| **何が足りないのか分からない** | `python scripts/check_prereqs.py` … 必須と任意を分けて出します |
| `brew: command not found` | Apple Silicon では PATH を通す必要があります（1 章） |
| `pip install` がコンパイルで止まる | Xcode Command Line Tools がありません（`xcode-select --install`・1 章） |
| 会議のあとの仕上げが「ffmpeg が要ります」で止まる | `brew install ffmpeg`（1 章） |
| 相手側がずっと無音 | 画面収録の許可（3 章）。許可したら**アプリごと再起動** |
| 「会議を始められませんでした」「『Python』を許可してください」 | `.app` から使うときの許可先は Python です（3 章）。許可したら会議アシスタントを終了して起動し直す |
| 自分の声が二重に出る | スピーカーで聞いている。ヘッドセットにする |
| 文字起こしがどんどん遅れる | ほかの GPU を使うアプリ（別の文字起こしアプリなど）を止める。または外（Deepgram・Gemini）へ |
| 「外へ出す」を選んだのに手元で動く | `gcloud auth login` が切れています（起動時の画面に理由が出ます） |
| 要約が出ない | `ollama serve` が動いているか。モデル名が設定と一致しているか |
| 議事録ではなく `minutes_prompt.md` ができた | 作る手段が無かった（Claude CLI・承認した Gemini・Ollama のどれも使えなかった）。中身を AI チャットに貼るか、6-1 のどれかを用意して `make_minutes.py` をもう一度 |
| 🔍 が薄く表示されて押せない | 裏取りの手段が無い。ボタンに乗せると理由が出ます（6-1） |
| 同じ人に複数の名前が付く | 画面の行から名前を付けると、以後と過去に反映されます |
| 「ネットワークへのアクセスを許可しますか」と何度も出る | macOS のファイアウォールです。**許可しなくても動きます**（3 章） |
| メニューバーに 🎙 が出ない | メニューバーが埋まっています。常駐アプリを 1 つ減らすか整理ツールで隠す（6 章） |
| Deepgram を選んだのに手元で動く | キーのファイルが無い／opt out の確認が通らなかった。起動時の画面に理由が出ます |

## 10. 自分の道具につなぐ（議事録・タスクの行き先）

**この配布版は、会議の中だけで完結します。** 出来たものは
`workspace/sessions/<会議>/` にファイルとして残るだけで、どこにも送りません。

一方、**作った側（自社）では、同じコードを社内の道具につないで使っています** —
議事録を Notion の会議録へ、TODO をタスク管理へ、辞書を会社ごとの用語集へ、
終わったことをチャットへ。その**つなぎ口はこの配布版にも残してあります**。
ただし相手側（どの Notion か、どのチャットか）は環境ごとに違うので、**中身は空にしてあります**。

| 口 | 場所 | 既定 | 何をする所か |
|---|---|---|---|
| 議事録の送り先 | `src/task_hub_minutes.py` | 使わない | 出来た `minutes.md` を、自分の置き場へ登録する |
| 話題ごとの連鎖 | `src/task_hub_chains.py` | 使わない | 同じ相手の前回の決定事項を引いて、今回とつなぐ |
| TODO の払い出し | `task_hub.notion_tasks` | `false` | 会議で出た TODO を、タスク管理へ登録する |
| チャット通知 | `task_hub.chat_notify_url` | 空 | 終わったことを Webhook で流す（Slack・Teams・その他） |
| 用語辞書の共有 | `task_hub.dictionary_sync` | `false` | 会社ごとの用語集を読み書きする |
| 名簿 | `task_hub.people_master_db` | 空 | 話者ラベルから人を引く台帳の場所 |
| 会議のあとに走らせるもの | `dispatch.on_finish` | `minutes` | `minutes`（議事録まで）／`none`（材料だけ） |

**組み替え方**は、次のどれでも構いません。

1. **口の中身を書き換える** … `src/task_hub_minutes.py` の登録先を、自分の Notion / Obsidian /
   Google ドライブ / 社内 API に差し替える。呼ばれる形（引数と戻り）はそのままで動きます。
   **中の実装は、作った側の社内ライブラリを読み込む形のまま**置いてあります（`~/.claude/skills/…`）。
   そのままでは動きません（無ければ、押しても「使えません」と出るだけで、会議には影響しません）。
   **動く例としてではなく、どこに何を書けばよいかの見本として**読んでください
2. **Webhook だけ使う** … `task_hub.chat_notify_url` に自分の受け口を書けば、
   コードを触らずに「終わりました」を流せます
3. **ファイルを拾う** … いちばん簡単です。`workspace/sessions/<会議>/` に
   `minutes.md`・`transcripts_final.jsonl`・`minutes_handoff.json`（決定事項と TODO を構造化したもの）が
   出るので、あとは自分のスクリプトで好きな所へ運べます

**鍵の置き方だけは、この道具と揃えることをおすすめします**（7 章）。設定に書くのはファイルの場所だけにして、
鍵の実体はリポジトリの外（`~/.config/meeting-copilot/`・`chmod 600`）に置いてください。

## 11. もっと詳しく

- 何がどこへ行くか、外部送信の関門の設計 → [SECURITY.md](SECURITY.md)
- 精度の測り方（自分の会議で試す） → `scripts/score_transcript.py` の冒頭
- 録画ファイルで試す → `python scripts/replay_meeting.py <動画かwav>`
