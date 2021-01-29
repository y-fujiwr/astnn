$dataset = @('roy','sesame','csn','gcj')
foreach($p in $dataset){
    python pipeline.py --lang java -s -c $p
}