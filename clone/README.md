y-fujiwrがASTNNに追加したもの  

# スクリプト  
### calculate_alltype_precision.py
ASTNNでスクリプト上に出力される精度は実際のものではない（非クローンペアが重複する）  
引数に，検出ログ（Type${n}.csv）が含まれるディレクトリを指定すると，実際の検出精度を計算してくれる

### cross_project.py
あるデータセットのモデルに別のデータセットのコード片を入力して検出精度を調査するためのスクリプト  
ハードコーディングで申し訳ありませんが，  
52行目付近の test_data に，各データセットからpipeline.pyにて作成される test/block.pkl をロード  
73行目付近の，model.load_state_dict(torch.load() のloadの引数に使用したいモデルのパスを指定  
83行目付近のresultdirにログを出力したいパスを指定  
すれば，動くと思います．

# データセット
### deepsimのGCJデータセット

1. deepsimデータセット <https://github.com/parasol-aser/deepsim/tree/master/dataset>  のgooglejam4.tar.gzを展開してください．
2. java -jar extractProcessingPart.jar -d ${1.のディレクトリ}  
    オプション -r を指定すると，例外処理のcatch,finallyブロック・Scanner,inputといったプロコン特有の入力受付文が削除されます．  
    1.のディレクトリ内に便宜的にメソッド名を拡張子としたファイルが出力されます．  
3. インライン展開は手動でお願いします．私は手動でやりました．  
4. python preprocess_gcj.py -d ${1.のディレクトリ}  
    astnn/clone/data 内に gcj というディレクトリが作成され，中に，gcj_funcs_all_no_inout.csv と gcj_pair_ids.pkl が作成されます．

### SeSaMeデータセット

required: .db 形式のファイルを読み込めてcsvに変換できるツール（私は DB Browser for SQLite を使用）  
1. SeSaMe <https://github.com/FAU-Inf2/sesame> リポジトリをクローン  
2. sesame/src で make を実行  
    リポジトリのクローンが始まり，最終的にdocs.dbが出力されます  
3. docs.dbを開き，テーブルinternal_filtered_methoddocsをcsv形式で出力  
4. preprocess_sesame.py  
    結構な時間がかかります  
    sesame_funcs_all.csv と sesame_pair_ids.pkl が作成されます．   
※オーバーロードなどで同名メソッドが複数存在する場合にメソッドをうまく抽出できていないという問題があります．  
私はクローンデータセットに使用されたメソッドの内，同名メソッドが服する存在するものを確認して手動で修正しました．  

### CodeSearchNetデータセット

required: Docker
1. CodeSearchNet <https://github.com/github/CodeSearchNet> リポジトリをクローン  
2. script/setup  
3. script/console  
    コンソールが立ち上がります．  
4. python train.py --model neuralbow  
    モデルの学習がスタート  
5. python predict.py -m ../resources/saved_models/${保存されたモデルの名前}  
    検索が始まります．結構な時間がかかります．resources/model_predictions.csv が作成されます．  
6. コンソールを終了し，resourcesディレクトリ内の preprocess_csn.py を実行  
    csn_funcs_all.csv と csn_pair_ids.pkl が作成されます．  

### Semantic dataset

1. Semantic Benchmark.zip <https://drive.google.com/file/d/1KicfslV02p6GDPPBjZHNlmiXk-9IoGWl/view> をダウンロード，解凍  
    論文：Farouq, Al-omari, Chanchal K. Roy, and Tonghao Chen, SemanticCloneBench: A Semanti Code Clone Benchmark using Crowd-Source Knowledge, Proc. of IWSC 2020, pp.57-63, London, ON, Canada, Feb. 2020.
2. python preprocess_roy.py -d Semantic\ Benchmark/Java/Stand\ Alone\ Clones  
    roy_funcs_all.csv と roy_pair_ids.pkl が作成されます．  
