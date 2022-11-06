# Usage

To run the script execute the ```run.sh``` file.

# 安装kaldi

1. 修改kaldi的原路径，将`Federated-learning-ASR/egs/kaldi-fl/pytorch-kaldi-fl/path.sh`中的`KALDI_ROOT`修改为自己安装的`kaldi`的路径

```shell
export KALDI_ROOT=/home/<user>/kaldi
```

2. 下载一些`librispeech`语言模型相关的一些[文件](http://openslr.magicdatatech.com/resources/11/)；

```shell
cd Federated-learning-ASR/egs/kaldi-fl/pytorch-kaldi-fl
mkdir data/local/lm
```

3. 执行 `run.sh`


4. 修改的部分：


   -  Stage 6 提取`MFCC`特征，原始代码为:

     ```shell
     if [ $stage -le 6 ]; then
       for part in `ls data/${set}`; do
         echo "Extracting mfcc and cmvn stats of pretrain set (${part})..."
         #sorting files
         utils/fix_data_dir.sh data/${set}/${part}    
         utils/utt2spk_to_spk2utt.pl data/${set}/${part}/utt2spk > data/${set}/${part}/spk2utt
         steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/${set}/${part} exp/make_mfcc/${set}/${part} $mfccdir/${set}/${part}
         steps/compute_cmvn_stats.sh data/${set}/${part} exp/make_mfcc/${set}/${part} $mfccdir/${set}/${part}
       done
     fi
     ```

     修改后的代码为：

     ```shell
     if [ $stage -le 6 ]; then
       for part in `ls data/whole`; do
         echo "Extracting mfcc and cmvn stats of pretrain set (${part})..."
         #sorting files
         utils/fix_data_dir.sh data/whole/${part}    
         utils/utt2spk_to_spk2utt.pl data/whole/${part}/utt2spk > data/whole/${part}/spk2utt
         steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/whole/${part} exp/make_mfcc/whole/${part} $mfccdir/whole/${part}
         steps/compute_cmvn_stats.sh data/whole/${part} exp/make_mfcc/whole/${part} $mfccdir/whole/${part}
       done
       
       for part in `ls data/pretrain`; do
         echo "Extracting mfcc and cmvn stats of pretrain set (${part})..."
         #sorting files
         utils/fix_data_dir.sh data/pretrain/${part}    
         utils/utt2spk_to_spk2utt.pl data/pretrain/${part}/utt2spk > data/pretrain/${part}/spk2utt
         steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/pretrain/${part} exp/make_mfcc/pretrain/${part} $mfccdir/pretrain/${part}
         steps/compute_cmvn_stats.sh data/pretrain/${part} exp/make_mfcc/pretrain/${part} $mfccdir/pretrain/${part}
       done
     fi
     ```

     不考虑切是`whole`或者`pretrain`，把需要的特征文件先都提了。

   |             | unseen        | federated                     | initial                        |
   | ----------- | ------------- | ----------------------------- | ------------------------------ |
   | **DIR**     | pretrain/test | pretrain/transfer_fl_test_org | pretrain/transfer_pre_test_org |
   | **#Nums**   | 476           | 682                           | 487                            |
   | **Initial** | 18.65         | 17.05                         | 14.39                          |
   | fl          | 19.32         | 18.17                         | 14.15                          |
   | ck2         | 19.37         | 17.89                         | 14.26                          |
   | ck4         | 19.36         | 17.88                         | 14.18                          |
   | ck8         | 19.43         | 17.93                         | 14.21                          |
   | ck16        | 19.48         | 18.01                         | 14.28                          |
   |             |               |                               |                                |

   

   