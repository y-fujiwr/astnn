$dataset = @('roy','sesame','csn','gcj')
foreach($p in $dataset){
    python generate_realdataset.py $p
}