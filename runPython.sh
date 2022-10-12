basep=./data/nerf/toyota4
savep=/testcase/
caseN=6
configN=customAll
tsteps=35000

if [[ -d $basep$savep$caseN ]]
then
    echo "$basep$savep$caseN exists on your filesystem."
else
    mkdir -v -p $basep$savep$caseN
fi

if [ $1 == "t" ]
then
    echo "Training data"
    
    python3 ./scripts/run.py --mode nerf \
            --scene $basep \
            --n_steps $tsteps \
            --network $configN.json \
            --save_snapshot $basep$savep$caseN/$configN-$tsteps.msgpack \
            --screenshot_transforms $basep/transforms.json \
            --screenshot_frames 4 \
            --screenshot_dir $basep$savep$caseN/ \
            --depth \
            |& tee $basep$savep$caseN/$configN-$tsteps-tlog.txt
fi

if [ $1 == "r" ]
then
    echo "Rendering data"
    python3 ./scripts/run.py --mode nerf \
            --scene $basep \
            --network $configN.json \
            --load_snapshot $basep$savep$caseN/$configN-$tsteps.msgpack \
            --screenshot_transforms $basep/transforms.json \
            --screenshot_frames 4 \
            --screenshot_dir $basep$savep$caseN/ \
            --depth \
            |& tee $basep$savep$caseN/$configN-$tsteps-rlog.txt
fi