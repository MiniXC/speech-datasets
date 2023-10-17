cd data
ROOT_DIR=/root/libritts-phones-and-mel/data
echo $ROOT_DIR
tar -czf $ROOT_DIR/dev_clean.tar.gz dev_clean
tar -czf $ROOT_DIR/dev_other.tar.gz dev_other
tar -czf $ROOT_DIR/test_clean.tar.gz test_clean
tar -czf $ROOT_DIR/test_other.tar.gz test_other
tar -czf $ROOT_DIR/train_clean_100.tar.gz train_clean_100
tar -czf $ROOT_DIR/train_clean_360.tar.gz train_clean_360
tar -czf $ROOT_DIR/train_other_500.tar.gz train_other_500