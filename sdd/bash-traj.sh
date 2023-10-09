for i in nexus_4; do
#$(find ./params/*); do
    for forecaster in darts; do
        filename=$(basename "$i")
        scene=$(echo "$filename" | rev | cut -d. -f2- | rev)
        for lr in 0.0 50 100 500 1000; do
            python plan_trajectory.py $scene $forecaster conformal\ controller $lr;
            python make_results.py $scene $forecaster conformal\ controller $lr;
        done
        for lr in 0.0 0.01 0.05 0.1; do
            python plan_trajectory.py $scene $forecaster aci $lr;
            python make_results.py $scene $forecaster aci $lr;
        done
        python plan_trajectory.py $scene $forecaster conservative 0.0;
        python make_results.py $scene $forecaster conservative 0.0;
    done
done
