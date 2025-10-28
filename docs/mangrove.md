## Mangrove Extent Mapping

OlmoEarth-v1-FT-Mangrove-Base is a model fine-tuned from OlmoEarth-v1-Base for preddicting mangrove extent from Sentinel-2.

Here are relevant links for fine-tuning and applying the model per the documentation in
[the main README](../README.md):

- Model checkpoint: https://huggingface.co/allenai/OlmoEarth-v1-FT-Mangrove-Base/resolve/main/model.ckpt
- rslearn dataset: https://huggingface.co/allenai/olmoearth_projects_mangrove/resolve/main/mangrove.tar

## Model Details

The model inputs twelve timesteps of satellite image data with one
mosaic [Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)
mosaic per 30-day period.

At every 2 by 2 patch it outputs a classifcication of mangrove, water or other

## Training Data

The model is trained on the
-what is the datast
- what is a sample composed of
- what is the rslearn window composed of
- how did we split the data

We split the dataset into train, val, and test splits spatially, where 128x128 pixel
grid cells are assigned via hash to train (70%), val (20%), or test (10%).

## Inference

Inference is documented in [the main README](../README.md). The prediction request
geometry should have start timestamp equal to the timestamp for which you want to make
the LFMC prediction (e.g., the current timestamp). The end timestamp won't be used and
can be set arbitrarily, e.g. set equal to the start timestamp.