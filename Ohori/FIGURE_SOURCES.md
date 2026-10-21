# 発表スライドの図: 元データ・件数・コード

発表スライド(全19枚)のグラフと例の数字は、すべて `analysis/31_slide_figures.py` がデータから直接計算している。
- 図の右下に「出典: Ohori/analysis/31_slide_figures.py」と入っている。
- 図と例に使った数字は、すべて `summary_tables/31_slide_numbers.csv` に出る。これまでの分析の表(19・20・21・22・25・26・27・28)と同じ値になることを確認した。
- 実行: `cd Ohori && python3 analysis/31_slide_figures.py`(約1分。scikit-learn・lifelines は不要。AUCは順位から、Kaplan–Meier法は自前で計算)
- 画面イメージ(19ページ)だけはHTMLで作った絵。右側の数字(3,000件・5日・26日・19%)も31番で計算し直して一致を確認した。
- 文字だけで出てくる件数(学習213,256件=約21万件、国分寺市22,332件、契約日14件)も31番が数える(31_slide_numbers.csv の「件数」)。
- 作り直す順番とデータの入手は `ROOM_LEVEL_ANALYSIS.md` の「発表スライドの図(29〜31番)」。

| ページ | 図 | 入力データ | 対象と件数 | 同じ計算をしたこれまでの分析 |
|---|---|---|---|---|
| 6 掲載日数 | figures/31a_duration.png | data/prepared/kokubunji_raw.parquet | 国分寺市 22,332件(2020〜2023年開始)。終わった割合はKaplan–Meier法 | 21_deck_numbers.py |
| 7 AUCとは | figures/31h_roc.png | results/17_fixed_window_features/candidates_2023.parquet | 2023年の候補 33,672件。AUC 0.625 | 30_roc_curve.py |
| 8〜9 確率の出し方の例 | (表と数字) | 同上 | 例A 1LDK・築10年未満・家賃8〜10万円 422件、7日以内19.6%(中央値)→1日平均3.1%。例B 1K・築30〜40年・家賃4〜6万円 2,301件、6.4%→0.9% | ― |
| 10 何日先か | figures/31b_horizon.png | 同上 | 同上 | 22_horizon_choice_figure.py |
| 12 2グループ版と3グループ版 | figures/31c1_levels_rates.png | candidates_2022 / 2023 / 2024q1 | 線は2022年 32,902件で決め、2023年 33,672件・2024年1〜3月 15,400件に当てた | 28_fair_three_vs_two.py |
| 13 AUCで比べて採用 | figures/31c2_levels_auc.png | 同上 | 同上 | 28_fair_three_vs_two.py, 27_three_vs_two_levels.py |
| 14 2グループの線 | figures/31g_boundary.png | candidates_2022 / 2023 | 同じ日・同じ市・同じ間取りの2件ずつの組 | 20_boundary_by_pairs.py |
| 15 2023年 | figures/31d_result2023.png | candidates_2023 | 33,672件。線は95%信頼区間 | 19_boundary_holdout.py |
| 16 偶然か、簡単な目安か | figures/31e_baselines.png | candidates_2022 / 2023 | 目安は2022年から作成。同じ部屋ごとの抽出し直し1000回(25と同じ乱数で、同じ値) | 25_overall_ci_baselines.py |
| 17 2024年1〜3月 | figures/31f_holdout.png | candidates_2023 / 2024q1 | 33,672件と15,400件 | 26_holdout_2024q1.py |
| 19 アプリ案 | figures/30_app_mockup.png(元: figures/30_app_mockup.html) | 画面はHTMLで作成。数字は summary_tables/24_same_condition_duration.csv | 1K・築10〜20年・6〜8万円は3,000件 | 24_same_condition_duration.py |

質問用の図(スライドには入れていない。同じスクリプトが出力): 31i 属性ごとの掲載日数、31j 線ごとの効き目と高の割合、31k 物件の似かたとAUC、31l 確率の確認、31m 情報を足したときのAUC。

## 候補データ(candidates_*.parquet)とは
- 確率の計算: 多摩地域、2021〜2022年に掲載開始の213,256件(約21万件)で、ロジスティック回帰により1日ごとに終わる確率を学び、7日分をつなげた(analysis/17, 26)。
- 候補: 年に5つの日付に掲載中だった物件。その日から7日以内に終わったか(target_7)と、7日以内に終わる確率(base_7)を持つ。

## よく聞かれそうな数字の違い
- 2024年の「高」の割合: 15%の線では58%。12〜13ページの比較用の線(11.7%)では74%。線が低いほど「高」が増える。
- 2グループの終了率: 12ページは比較用の線(11.7%)で7%と14%。15ページ以降は発表で使う15%の線で7.9%と15.4%。
- 2023年の平均の予測: 7日以内は12.5%(実際10.2%)。以前のスライドの14.5%は誤りで、データから計算し直すと12.5%。
- 2024年の3グループ版と2グループの差: 同じ探し方の2グループ版とは+0.013、発表で使う15%の線とは+0.004(以前の「0.003」は丸め誤り)。
