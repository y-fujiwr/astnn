$dataset = @('roy_bigram','sesame_bigram','csn_bigram','gcj_bigram')
foreach($p in $dataset){
    python train.py --lang java -r -m lstm -v monogram
    python cross_project.py --lang java -r -t $p -m lstm -v monogram
}