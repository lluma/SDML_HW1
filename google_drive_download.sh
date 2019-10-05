gdrive_download_zip () {
  file_id=$1
  output_filename=$2
  wget --no-check-certificate -r "https://docs.google.com/uc?export=download&id=$file_id" -O $output_filename
}

gdrive_download() {
  file_id=$1
  output_filename=$2
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$file_id" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $output_filename

  rm -rf /tmp/cookies.txt
}

echo 'create data folder'
mkdir data

echo 'Downloading some zipped files......'

echo 'Downloading pretrained embedding words from GloVe...'
gdrive_download 1quJXyFoy3jMuO9mHr1ze8xuMF2T_VXcw data/glove.6B.300d.txt

echo 'Downloading training set...'
gdrive_download_zip 1U0Ay9jzr4rQJGOjhvH7918NFAv3HyVev data/task2_trainset.zip

echo 'Downloading public testing set'
gdrive_download_zip 1tYLP3j1RrbcfOvL8QnMpz1blbq5EqqSY data/task2_public_testset.zip
