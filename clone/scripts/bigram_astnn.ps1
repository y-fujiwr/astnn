#python pipeline.py --lang java -v bigram_astnn
python train.py --lang java -r -v bigram_astnn -g
$dataset = @('roy','sesame','csn','gcj')
foreach($p in $dataset){
    python pipeline.py --lang java -v bigram_astnn -c $p -s
    python cross_project.py --lang java -r -t ${p}_bigram_astnn
}
