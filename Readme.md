
# Run the code

### Dependencies
* Python 3.6
* Run pip install -r requirements.txt

### Instructions
1. The datasets are available [here](https://drive.google.com/drive/u/8/folders/1ieF6mofbZ0F2qG6gYJKVYP3fa2ARMu54). Change the dataset path in run.sh data=/data_path
2. **sh run.sh** to run the code.

### Note
* Right now the code only supports single GPU training, but an extension to support multiple GPUs should be easy.
* The personalization, event descriptions encoding by SBERT is done in transformer/Models.py code.
* External Stimuli aware attention is done in transformer/SubLayers.py code.
* Under transformer/Constant.py exogenous event are defined for each dataset.
* There are several factors that can be changed, beside the ones in **run.sh**:
  * In **Utils.py**, function **log_likelihood**, users can select whether to use numerical integration or Monte Carlo integration.
  * In **transformer/Models.py**, class **Transformer**, there is an optional recurrent layer. This  is inspired by the fact that additional recurrent layers can better capture the sequential context, as suggested in [this paper](https://arxiv.org/pdf/1904.09408.pdf). In reality, this may or may not help, depending on the dataset.



