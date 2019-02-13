# All Graphs Lead To Rome
Graph Convolutional Networks for multi-image matching. For more technical details see
the ArXiv paper (https://arxiv.org/abs/1901.02078).

# Dependencies
All the code here is written in Python 3. You will need the following depencies:
* [TensorFlow GPU](https://www.tensorflow.org/install)
* [Sonnet](https://github.com/deepmind/sonnet)
* [NumPy](http://www.numpy.org/)
* [Matplotlib](https://matplotlib.org/users/installing.html)
* [SciPy](https://www.scipy.org/install.html)
* [Scikit-learn](https://scikit-learn.org/stable/install.html)
* [PyYaml](https://pyyaml.org/)
* [Pickle](https://docs.python.org/3/library/pickle.html)
* [Pillow](https://pillow.readthedocs.io/en/stable/)
* [TQDM](https://github.com/tqdm/tqdm)
* [Gzip](https://docs.python.org/3/library/gzip.html)
* [argparse](https://docs.python.org/3/library/argparse.html)
* [argcomplete](https://pypi.org/project/argcomplete/)

# Basic Code Use

## Dataset Generation
To generate the synthetic datasets, call the `data_util` module to generate it in
```
$ python3 -m data_util --dataset=noise_pairwise5
```
For generating the Rome16K datasets, you need to download and untar/unzip the [Rome16K dataset](http://www.cs.cornell.edu/projects/p2f/) in a directory you choose then specify that directory in the options as `rome16_dir`. To do the initial generation, specify where you save it in `top_dir`:
```
$ python3 -m data_util.rome16k --top_dir=/your/location/Rome16K
$ python3 -m data_util --dataset=rome16kgeom0

```
The synthetic datasets take around 4-10 GB, whereas the Rome16K datasets take around 200GB, so you will need space. If you don't have that space, you can specify using some of the options in `data_util` such as `max_num_tuples`

## Training and Testing
If the datasets are already generated, training can just be done by calling the ` train.py`. For example:
```
$ python3 train.py \
  --save_dir=save/testing \
  --dataset=noise_pairwise5view5 \
  --architecture=longskip2 \
  --loss_type=l1 \
  --geometric_loss=2 \
  --use_end_bias=false \
  --use_abs_value=true \
  --final_embedding_dim=80 \
  --learning_rate=1e-4 \
  --min_learning_rate=5e-7 \
  --learning_rate_continuous=True \
  --learning_rate_decay_type=exponential \
  --learning_rate_decay_epochs=3e-1 \
  --use_unsupervised_loss=true \
  --optimizer_type=adam \
  --load_data=true \
  --batch_size=8 \
  --train_time=55 \
  --test_freq=5 \
  --save_interval_secs=598 \
  # End Args
```
You can find the list of options for datasets and models using `python3 train.py --help`.

# Code Layout
The code has 3 basic components: train/test, data utilities, and models. All of these depend on the `options.py`, `myutils.py`, and `tfutils.py`, with `options.py` carrying around all the global parameters.

The data utilities (`data_util`) handles the generation, saving, and loading of the datasets. The dataset classes are based off `GraphSimDataset` in `parent_dataset.py`. It heavily uses tfrecords for fast loading at training/testing time. The output of the dataset is generally a dictionary with all the necessary parts, in our case the graph laplacian and initial embeddings as well as auxillary things such as the adjacency matrx of the graph and the ground truth embeddings for testing evaluation. Rome16K also includes geometric information between the views which can be used during training. Unfortunately, sparsity is not exploited right now so all the graph matrices are stored densly. Future work include sparisfying all these.

The `models` folder takes in the output of the datasets and then puts it through a [Sonnet](https://github.com/deepmind/sonnet) module based network. The modules all require as input the graph laplacian and the initial node embeddings, the sizes of which should be known in advance. The exact nature of the modules can be safely abstracted, so this is fairly modular.

The training and testing is fairly straightforward - once the dataset is generated and saved on disk and the model chosen, you specify the options of how you want to train it (as shown in the example above) and run it. The above example is an example of a typical run, and can be used as starting point. To test the baselines, you will need MATLAB - they are all in the `baselines` folder.


# Questions
If you have any questions, please ask me at stephi@seas.upenn.edu


