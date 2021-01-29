$models = @('astnn','lstm','bilstm','dnn')
foreach($m in $models){
    $dataset = @('roy_w2v_real','sesame_w2v_real','csn_w2v_real','gcj_w2v_real')
    foreach($p in $dataset){
        python cross_project.py --lang java -t $p -m $m -g
        python cross_project.py --lang java -t $p -m $m -g -r
    }
}