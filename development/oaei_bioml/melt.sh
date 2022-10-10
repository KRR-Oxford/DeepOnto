#!/bin/bash
set -e

data_source=$1
shift
rel=equiv
data_dir=/home/yuan/projects/DeepOnto/data/${data_source}/${rel}_match
tool_name=$1
tool_path=$2
src=$3
tgt=$4
topic=$5

if [ -z "$topic" ]
  then
    echo "Evaluate on ${src}2${tgt}"
    src_onto=$data_dir/ontos/$src.owl
    tgt_onto=$data_dir/ontos/$tgt.owl
    ref_dir=$data_dir/refs/${src}2${tgt}
    result_path=${tool_name}/${src}2${tgt}/melt
  else
    echo "Evalute on ${src}2${tgt}.${topic}"
    src_onto=$data_dir/ontos/${src}.${topic}.owl
    tgt_onto=$data_dir/ontos/${tgt}.${topic}.owl
    ref_dir=$data_dir/refs/${src}2${tgt}.${topic}
    result_path=${tool_name}/${src}2${tgt}.${topic}/melt
fi

exp_dir=$result_path/../
mkdir -p $result_path
client=matching-eval-client-latest.jar
# tool_path=logmap-lite-melt-oaei-2021-web-latest.tar.gz

# sudo java -Xms500M -Xmx15G -DentityExpansionLimit=100000000 -jar $client -s $tool_path --local-testcase $src_onto $tgt_onto test.rdf -r $result_path 2> $result_path/err.log > $result_path/out.log
# need to copy the alignment.rdf out
# sudo mv $result_path/LocalTrack_1.0/Local+TC/*/systemAlignment.rdf $result_path/systemAlignment.rdf
echo $exp_dir
python read_oaei.py -r $result_path

python /home/yuan/projects/DeepOnto/om_eval.py -o $exp_dir/ -p $exp_dir/src2tgt -r $ref_dir/unsupervised/test.tsv -n $ref_dir/unsupervised/val.tsv
sudo mv $exp_dir/global_match.results.json $exp_dir/us.equiv.results.json
python /home/yuan/projects/DeepOnto/om_eval.py -o $exp_dir -p $exp_dir/src2tgt -r $ref_dir/semi_supervised/test.tsv -n $ref_dir/semi_supervised/train+val.tsv
sudo mv $exp_dir/global_match.results.json $exp_dir/ss.equiv.results.json

python print_html.py -t $tool_name -u $exp_dir/us.equiv.results.json -s $exp_dir/ss.equiv.results.json