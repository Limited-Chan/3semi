# GitHub 再構築手順

履歴は残し、現在のtrackedファイルだけを新しい分析一式へ置き換える。

## 推奨: 新しくcloneして作業

```powershell
cd "$env:USERPROFILE\Downloads"
git clone https://github.com/Limited-Chan/3semi.git 3semi_rebuild
cd .\3semi_rebuild
git switch main
git pull origin main
git switch -c rebuild-equal-bins-v2.3
```

## 現在Git管理されている内容をブランチ上で削除

```powershell
git rm -r .
```

これは `.git` 履歴を削除しない。trackedファイルの削除をコミット候補にするだけ。

## v2.3をコピー

ZIPを `C:\Users\banan\Downloads\3semi_equal_bins_v2_3` に展開した例:

```powershell
$src = "C:\Users\banan\Downloads\3semi_equal_bins_v2_3\3semi_equal_bins_v2_3"
Get-ChildItem $src -Force | Copy-Item -Destination . -Recurse -Force
```

展開時に二重フォルダになっていなければ `$src` を1階層浅くする。

## テスト

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v test_analysis.py
```

## Gitに載るものを確認

```powershell
git status
git diff --stat
git diff -- .gitignore README.md METHODOLOGY.md analyze.py test_analysis.py
```

元TSV、prepared、results、.venv が `git status` に出ていないことを確認する。

## commit / push

```powershell
git add -A
git status
git commit -m "Rebuild analysis with full-range equal-width bins"
git push -u origin rebuild-equal-bins-v2.3
```

GitHubで `rebuild-equal-bins-v2.3` → `main` のPRを作成し、差分を確認してからmergeする。
