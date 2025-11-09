# SpikeATE
# ACM Transactions on Knowledge Discovery from Data 2025: Efficient Aspect Term Extraction using Spiking Neural Network

## Envrionment

* Python 3.10.12
* Numpy 1.26.4
* Spacy 3.7.5
* Fasttext 0.9.3
* Torch 2.4.1+cu121
* GPU : Tesla T4


### Steps to run code

1. To download SemEval-2014, 2015, and 2016 datasets, click here, [SemEval-2014][1], [SemEval-2015][2], [SemEval-2016][3]
2. To download [Yelp review][4] and [Amazon Cell phones and accessories][5] datasets. 
3. To download the [glove][6] and [fasttext][7] embeddings.
4. Then, use DataProcessing.ipynb file to preprocess the datasets. 


## Argument Descriptions

* --epoch: Number of training epochs. 

* --modelType: Specifies the model architecture to use. "HypoxSpike" refers to the ternary/binary SNN model.

* --learningRate: Learning rate ($\eta$) for the optimizer. 

* --resetMechanism: Mechanism to reset membrane potential after a spike. Can be "Hard" or "soft" or "no" reset.

* --spkNature: Defines the type of spike representation. Can be "binary" or "ternary".

  * "binary" allows only 0 and 1 spikes.
  * "ternary" allows -1, 0, and 1 spikes.

* --vTh: Threshold voltage $V_{thr}$ used in spiking neuron activation. 

* --scDecay: Synaptic current decay rate. Controls how fast the synaptic current decays over time. 

* --vDecay: Membrane voltage decay rate. Controls how fast the membrane potential decays over time.

* --numTimeSteps: Simulation time steps per input sample. 

* --batchSize: Number of samples processed in one forward/backward pass.

* --nCls: Number of output classes. For aspect term classification, this is 3 (B, I, O).

* --isadapVth: Flag to enable adaptive threshold voltage in the spiking neuron model. Set 1 to enable adaptive threshold; 0 for fixed threshold.

* --seqLen: Sequence length (number of tokens per sentence). 

* --preembeddingType: Type of pre-trained word embedding to use: {glove, fasttext, word2vec}. 
     
* --embedDim: Word embedding dimension. Defines the size of each word (token) vector (e.g., 300 for GloVe). 

* --numChn: Number of output channels in the convolutional spiking layer. 

* --numValSamples: Number of samples held out from training for validation and hyperparameter tuning. 

* --trainDataPath: Path to the training dataset file. 

* --testDataPath: Path to the test dataset file. 

* --seedNum: Random seed for reproducibility. 
             
* --numRuns: Number of independent runs of the experiment. 
                         

```
train_data_path = '/SemEval/SemEval14/Laptops_Train.xml'
test_data_path = '/SemEval/SemEval14/Laptops_Test_Data_phaseB.xml'
python main.py --trainDataPath {train_data_path} --testDataPath {test_data_path} --batchSize 8 --seedNum 43 --epoch 22 --resetMechanism "zero" --spkNature "ternary" --numTimeSteps 6 --seqLen 80 --vTh 0.1 --scDecay 0.1 --vDecay 0.1 --nCls 3 
```
```
train_data_path = '/SemEval/SemEval14/Restaurants_Train.xml'
test_data_path = '/SemEval/SemEval14/Restaurants_Test_Data_phaseB.xml'
python main.py --trainDataPath {train_data_path} --testDataPath {test_data_path} --batchSize 8 --seedNum 43 --epoch 20 --resetMechanism "zero" --spkNature "ternary" --numTimeSteps 6 --seqLen 82 --vTh 0.1 --scDecay 0.1 --vDecay 0.1 --nCls 3 
```
```
train_data_path = '/SemEval/SemEval15/ABSA-15_Restaurants_Train_Final.xml'
test_data_path = '/SemEval/SemEval15/ABSA15_Restaurants_Test.xml'
python main.py --trainDataPath {train_data_path} --testDataPath {test_data_path} --batchSize 8 --seedNum 43 --epoch 18 --resetMechanism "zero" --spkNature "ternary" --numTimeSteps 6 --seqLen 76 --vTh 0.1 --scDecay 0.1 --vDecay 0.1 --nCls 3 
```
```
train_data_path = '/SemEval/SemEval16/ABSA16_Restaurants_Train_SB1_v2.xml'
test_data_path = '/SemEval/SemEval16/EN_REST_SB1_TEST.xml.gold'
python main.py --trainDataPath {train_data_path} --testDataPath {test_data_path} --batchSize 8 --seedNum 43 --epoch 22 --resetMechanism "zero" --spkNature "ternary" --numTimeSteps 6 --seqLen 78 --vTh 0.1 --scDecay 0.1 --vDecay 0.1 --nCls 3 
```

### Convolutional spike encoding operation for the first two time steps:

![](SpkEnc_SpikeATE.png)



[1]: https://alt.qcri.org/semeval2014/task4/
[2]: https://alt.qcri.org/semeval2015/task12/
[3]: https://alt.qcri.org/semeval2016/task5/
[4]: https://www.yelp.com/dataset
[5]: https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews
[6]: https://nlp.stanford.edu/projects/glove/
[7]: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
