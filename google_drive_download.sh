gdrive_download () {
  file_id=$1
  output_filename=$2
  wget --no-check-certificate -r "https://docs.google.com/uc?export=download&id=$file_id" -O $output_filename
}

echo 'create data folder'
mkdir data

echo 'Downloading some zipped files......'

echo 'Downloading pretrained embedding words from GloVe...'
gdrive_download 1quJXyFoy3jMuO9mHr1ze8xuMF2T_VXcw data/glove.6B.zip

echo 'Downloading training set...'
gdrive_download 1U0Ay9jzr4rQJGOjhvH7918NFAv3HyVev data/task2_trainset.zip

echo 'Downloading public testing set'
gdrive_downloading 1tYLP3j1RrbcfOvL8QnMpz1blbq5EqqSY data/task2_public_testset.zip
