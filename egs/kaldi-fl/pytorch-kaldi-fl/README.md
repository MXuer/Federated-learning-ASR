# Usage

To run the script execute the ```run.sh``` file.



# 环境

根据[pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi)的`readme`中所述，摘抄如下：

```text
If not already done, install PyTorch (http://pytorch.org/). We tested our codes on PyTorch 1.0 and PyTorch 0.4. An older version of PyTorch is likely to raise errors. To check your installation, type “python” and, once entered into the console, type “import torch”, and make sure no errors appear.

We recommend running the code on a GPU machine. Make sure that the CUDA libraries (https://developer.nvidia.com/cuda-downloads) are installed and correctly working. We tested our system on Cuda 9.0, 9.1 and 8.0. Make sure that python is installed (the code is tested with python 2.7 and python 3.7). Even though not mandatory, we suggest using Anaconda (https://anaconda.org/anaconda/python).
```

提取出来需要的信息就是：

| 环境    | 信息    |
| ------- | ------- |
| kaldi   | -       |
| pytorch | 1.0/0.4 |
| python  | 3.7     |
| cuda    | 9.0/8.0 |
| linux   | ubuntu  |

因为`librispeech`的原始语音为`flac`格式，所以需要安装`flac`相关的库；

```shell
apt-get install flac
```

# 实验步骤

1. 修改kaldi的原路径，将`Federated-learning-ASR/egs/kaldi-fl/pytorch-kaldi-fl/path.sh`中的`KALDI_ROOT`修改为自己安装的`kaldi`的路径

   ```shell
   export KALDI_ROOT=/home/<user>/kaldi
   ```

2. 下载一些`librispeech`语言模型相关的一些[文件](http://openslr.magicdatatech.com/resources/11/)；（此处可忽略，因为`run.sh`中`Stage 1`中的`locale/download_lm.sh`也会自动下载；如果下载的网速太慢，可以自己先下载到Windows上，然后上传到`Federated-learning-ASR/egs/kaldi-fl/pytorch-kaldi-fl/data/local/lm`中，如果该文件夹不存在，请按照下列指令创建。）可以尝试先直接运行`./run.sh`，看下网速，如果不行再下载到本地。

   ```shell
   cd Federated-learning-ASR/egs/kaldi-fl/pytorch-kaldi-fl
   mkdir data/local/lm
   ```

   需要下载的文件及地址见下表：

   | 文件名                                                       |
   | ------------------------------------------------------------ |
   | [4-gram.arpa.gz](http://openslr.magicdatatech.com/resources/11/4-gram.arpa.gz) |
   | [g2p-model-5](http://openslr.magicdatatech.com/resources/11/g2p-model-5) |
   | [librispeech-lexicon.txt](http://openslr.magicdatatech.com/resources/11/librispeech-lexicon.txt) |
   | [librispeech-lm-corpus.tgz](http://openslr.magicdatatech.com/resources/11/librispeech-lm-corpus.tgz) |
   | [librispeech-lm-norm.txt.gz](http://openslr.magicdatatech.com/resources/11/librispeech-lm-norm.txt.gz) |
   | [librispeech-vocab.txt](http://openslr.magicdatatech.com/resources/11/librispeech-vocab.txt) |

3. 执行 `run.sh`；【此处为训练`whole`模型，此时代码22行中的`set`为`whole`】

   ```shell
   cd Federated-learning-ASR/egs/kaldi-fl/pytorch-kaldi-fl
   chmd 775 -R .
   ./run.sh
   ```


4. 修改`run.sh`中22行的`set`为`pretrain`，26行的`stage`为`7`；

5. 执行`run.sh`；

6. 运行完成之后，获取所有实验的结果的`wer`；

   ```shell
   cd Federated-learning-ASR/egs/kaldi-fl/pytorch-kaldi/exp
   find . -name best_wer | xargs tail
   ```

7. 实验结果：

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

8. 代码中修改的部分：


   -  `run.sh`中`stage 6 `提取`MFCC`特征，原始代码为:

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

   -  `run.sh`中，当`set=pretrain`时，`stage 18`修改较多，原始代码陷入了死循环，`while`后面所接的判断条件中的`$spk`并未赋值，导致了这个判断条件永远都是`true`，程序就会无休止的运行下去。本步骤是不同数量的`client`的联邦学习，根据论文中的算法，所以去掉这个死循环就好了。原始代码为：

      ```shell
      if [ $stage -le 18 ]; then
          echo "Training on the FL set"
          cd ../pytorch-kaldi
          for ck in 4 16 8 2; do
              
              while ! [ -f exp/fl/ck${ck}/clients/fl_${spk}_mlp_mfcc/exp_files/final_architecture1.pkl ]; do
                  echo "exp/fl/ck${ck}/clients/fl_${spk}_mlp_mfcc/exp_files/final_architecture1.pkl"
                  for spk in `ls ../pytorch-kaldi-fl/data/FL | sort -n` ; do #
                      if [ "${spk}" -ne "3559" ]; then
                         mkdir -p exp/fl/ck${ck}/clients
                         # if this crashes on the first try (depends on the system) 
                         # you need to generate this file by hand and restart this stage:
                         # echo "0" > exp/fl/ck${ck}/clients/last_spk
                         touch exp/fl/ck${ck}/clients/last_spk
                         lspk=$(cat exp/fl/ck${ck}/clients/last_spk)
                         if [ "${spk}" -gt "$lspk" ]; then
                         #mkdir -p exp/fl/ck${ck}/clients/fl_${spk}_mlp_mfcc
                         until python run_exp_fl.py cfg/fl/${ck}/fl_${spk}_mlp_mfcc.cfg
                         do
                            echo "some error occured in spk ${spk}... retrying " >> main.log
                            sleep 1
                         done
                          echo "${spk}" > exp/fl/ck${ck}/clients/last_spk
                         fi
                      fi
                  done
                  
                  echo "0" > exp/fl/ck${ck}/clients/last_spk
                  #wait
                  # average here
                  cd ../pytorch-kaldi-fl
                  python local/weighted_avg.py ../pytorch-kaldi/exp/fl/ck${ck} data/FL
                  cd ../pytorch-kaldi
              done
          done
      
      fi
      ```

      修改后代码为：

      ```shell
      if [ $stage -le 18 ]; then
          echo "Training on the FL set"
          cd ../pytorch-kaldi
          for ck in 2 4 8 16; do
            for spk in `ls ../pytorch-kaldi-fl/data/FL | sort -n` ; do #
                if [ "${spk}" -ne "3559" ]; then
                    mkdir -p exp/fl/ck${ck}/clients
                    # if this crashes on the first try (depends on the system) 
                    # you need to generate this file by hand and restart this stage:
                  #  echo "0" > exp/fl/ck${ck}/clients/last_spk
                    touch exp/fl/ck${ck}/clients/last_spk
                    lspk=$(cat exp/fl/ck${ck}/clients/last_spk)
                    if [ "${spk}" -gt "$lspk" ]; then
                      #mkdir -p exp/fl/ck${ck}/clients/fl_${spk}_mlp_mfcc
                      until python run_exp_fl.py cfg/fl/${ck}/fl_${spk}_mlp_mfcc.cfg
                      do
                          echo "some error occured in spk ${spk}... retrying " >> main.log
                          sleep 1
                      done
                      echo "${spk}" > exp/fl/ck${ck}/clients/last_spk
                    fi
                fi
            done
            
            echo "0" > exp/fl/ck${ck}/clients/last_spk
            #wait
            # average here
            cd ../pytorch-kaldi-fl
            python local/weighted_avg.py ../pytorch-kaldi/exp/fl/ck${ck} data/FL
            cd ../pytorch-kaldi
          done
      
      fi
      ```

   -  `run.sh`中所有的`local/score.sh`均修改成了`local/score_wer.sh`，这样可以直接输出每一个case中最好的`wer`；

   -  `Federated-learning-ASR/egs/kaldi-fl/pytorch-kaldi-fl/local/weighted_avg.py`修改的较多；根据论文和代码，本文件会生成后续步骤中需要用的`final_architecture1.pkl`，但是原始代码压根就没有生成这个文件。所以修改了一下，使得这个代码起作用；这个部分较多，具体请参考原代码和修改后的代码，使用`vimdiff <源文件地址> <修改后的地址>`即可直观查看前后文件修改的部分；

   -  配置文件的修改：有多个配置文件进行了修改，修改的配置文件均位于`Federated-learning-ASR/egs/kaldi-fl/pytorch-kaldi/cfg`，修改的部分有三种情况：


      -  配置文件所对应的训练任务，会导入前面步骤中的预训练模型，这些配置文件中设定的输出神经元的个数与预训练模型的输出神经元个数不匹配。所以做此修改。具体修改的文件及细节如下：

         ```shell
         ./fl/pretrain_mlp_mfcc_prod.cfg
         116c116
         < dnn_lay = 1024,1024,1024,1024,1640
         ---
         > dnn_lay = 1024,1024,1024,1024,1664
         
         ./fl/fl_16_mlp_mfcc_prod.cfg
         115c115
         < dnn_lay = 1024,1024,1024,1024,1640
         ---
         > dnn_lay = 1024,1024,1024,1024,1664
         
         
         ./fl/fl_2_mlp_mfcc_prod.cfg
         115c115
         < dnn_lay = 1024,1024,1024,1024,1640
         ---
         > dnn_lay = 1024,1024,1024,1024,1664
         
         
         ./fl/fl_8_mlp_mfcc_prod.cfg
         115c115
         < dnn_lay = 1024,1024,1024,1024,1640
         ---
         > dnn_lay = 1024,1024,1024,1024,1664
         
         
         115c115
         < dnn_lay = 1024,1024,1024,1024,1640
         ---
         > dnn_lay = 1024,1024,1024,1024,1664
         
         
         
         ./fl/whole_mlp_mfcc_prod.cfg
         116c116
         < dnn_lay = 1024,1024,1024,1024,1656
         ---
         > dnn_lay = 1024,1024,1024,1024,1672
         ```

         

      -  配置文件中的路径不对；具体修改如下：

         ```shell
         ./whole/mlp_mfcc_prod.cfg
         18,19c18,19
         < 	fea_lst=../pytorch-kaldi-whole/data/whole/train/feats.scp
         < 	fea_opts=apply-cmvn --utt2spk=ark:../pytorch-kaldi-whole/data/whole/train/utt2spk  ark:../pytorch-kaldi-whole/mfcc/whole/train/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
         ---
         > 	fea_lst=../pytorch-kaldi-fl/data/whole/train/feats.scp
         > 	fea_opts=apply-cmvn --utt2spk=ark:../pytorch-kaldi-fl/data/whole/train/utt2spk  ark:../pytorch-kaldi-fl/mfcc/whole/train/cmvn_train.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
         25c25
         < 	lab_folder=../pytorch-kaldi-whole/exp/alignments/whole/train/
         ---
         > 	lab_folder=../pytorch-kaldi-fl/exp/alignments/whole/train/
         28,29c28,29
         < 	lab_data_folder=../pytorch-kaldi-whole/data/whole/train/
         < 	lab_graph=../pytorch-kaldi-whole/exp/whole/tri3/graph/
         ---
         > 	lab_data_folder=../pytorch-kaldi-fl/data/whole/train/
         > 	lab_graph=../pytorch-kaldi-fl/exp/whole/tri3/graph/
         37,38c37,38
         < 	fea_lst=../pytorch-kaldi-whole/data/pretrain/dev/feats.scp
         < 	fea_opts=apply-cmvn --utt2spk=ark:../pytorch-kaldi-whole/data/pretrain/dev/utt2spk  ark:../pytorch-kaldi-whole/mfcc/pretrain/dev/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
         ---
         > 	fea_lst=../pytorch-kaldi-fl/data/pretrain/dev/feats.scp
         > 	fea_opts=apply-cmvn --utt2spk=ark:../pytorch-kaldi-fl/data/pretrain/dev/utt2spk  ark:../pytorch-kaldi-fl/mfcc/pretrain/dev/cmvn_dev.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
         44c44
         < 	lab_folder=../pytorch-kaldi-whole/exp/alignments/pretrain/dev/
         ---
         > 	lab_folder=../pytorch-kaldi-fl/exp/alignments/pretrain/dev/
         47,48c47,48
         < 	lab_data_folder=../pytorch-kaldi-whole/data/pretrain/dev/
         < 	lab_graph=../pytorch-kaldi-whole/exp/whole/tri3/graph/
         ---
         > 	lab_data_folder=../pytorch-kaldi-fl/data/pretrain/dev/
         > 	lab_graph=../pytorch-kaldi-fl/exp/whole/tri3/graph/
         64c64
         < 	lab_graph=../pytorch-kaldi-whole/exp/whole/tri3/graph/
         ---
         > 	lab_graph=../pytorch-kaldi-fl/exp/whole/tri3/graph/
         70,71c70,71
         < 	fea_lst=../pytorch-kaldi-fl/data/pretrain/transfer_test/feats.scp
         < 	fea_opts=apply-cmvn --utt2spk=ark:../pytorch-kaldi-fl/data/pretrain/transfer_test/utt2spk  ark:../pytorch-kaldi-fl/mfcc/pretrain/transfer_test/cmvn_transfer_test.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
         ---
         > 	fea_lst=../pytorch-kaldi-fl/data/pretrain/transfer_fl_test_org/feats.scp
         > 	fea_opts=apply-cmvn --utt2spk=ark:../pytorch-kaldi-fl/data/pretrain/transfer_fl_test_org/utt2spk  ark:../pytorch-kaldi-fl/mfcc/pretrain/transfer_fl_test_org/cmvn_transfer_fl_test_org.ark ark:- ark:- | add-deltas --delta-order=2 ark:- ark:- |
         76,77c76,77
         < 	lab_data_folder=../pytorch-kaldi-fl/data/pretrain/transfer_test/
         < 	lab_graph=../pytorch-kaldi-whole/exp/whole/tri3/graph/
         ---
         > 	lab_data_folder=../pytorch-kaldi-fl/data/pretrain/transfer_fl_test_org/
         > 	lab_graph=../pytorch-kaldi-fl/exp/whole/tri3/graph/
         104c104
         < dnn_lay = 1024,1024,1024,1024,1648
         ---
         > dnn_lay = 1024,1024,1024,1024,1672
         129c129
         < normalize_with_counts_from = ../pytorch-kaldi-whole/exp/whole/tri3_ali/ali_train_pdf.counts
         ---
         > normalize_with_counts_from = ../pytorch-kaldi-fl/exp/whole/tri3_ali/ali_train_pdf.counts
         ```

      -  修改了配置文件中的`batch_size`；

         ```shell
         ./fl/whole_mlp_mfcc.cfg
         77c77
         < batch_size_train = 128
         ---
         > batch_size_train = 256
         ```

         

   