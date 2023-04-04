set -x

if [ ! -d "${DATA_PATH}" ]; then
	echo "please export the DATA_PATH first!"
	exit 1
fi


#convert dataset and model
pushd models
python save_bert_inference.py -m $DATA_PATH/model -o ../bert.pt
popd

pushd datasets
python save_squad_features.py -m $DATA_PATH/model -d $DATA_PATH/dataset -o ../squad.pt
popd

