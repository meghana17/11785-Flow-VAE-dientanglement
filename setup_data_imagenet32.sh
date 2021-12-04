cd ./DenseFlow
mkdir -p ./data/imagenet32/{raw,processed}
cp ./make_imagenet_data.py ./data/imagenet32/processed/
cd ./DenseFlow/data/imagenet32/raw
wget https://image-net.org/data/downsample/Imagenet32_train.zip
wget https://image-net.org/data/downsample/Imagenet32_val.zip
unzip Imagenet32_train.zip
unzip Imagenet32_val.zip
cd ../processed
mkdir -p ./{train_32x32,valid_32x32}
python make_imagenet_data.py --img_size 32
