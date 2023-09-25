from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

reduction_method = UMAP

plot_books = False

speaker_vectors = []
speaker_names = []

for speaker_file in Path("data").glob("**/*.npy"):
    if "_speaker.npy" not in speaker_file.name:
        continue
    if plot_books:
        speaker_name = speaker_file.parent.name
    else:
        speaker_name = speaker_file.parent.parent.name
    speaker_names.append(speaker_name)
    speaker_vectors.append(np.load(speaker_file))

speaker_vectors = np.array(speaker_vectors)
speaker_vectors = reduction_method(n_components=2).fit_transform(speaker_vectors)

plt.figure(figsize=(10, 10))
sns.scatterplot(
    x=speaker_vectors[:, 0],
    y=speaker_vectors[:, 1],
    hue=speaker_names,
    legend=False,
)
plt.savefig("figures/speaker_vectors.png")

speaker_vectors = []
speaker_names = []

for speaker_file in Path("data").glob("**/*.png"):
    if "_speaker.png" not in speaker_file.name:
        continue
    if plot_books:
        speaker_name = speaker_file.parent.name
    else:
        speaker_name = speaker_file.parent.parent.name
    speaker_names.append(speaker_name)
    img_mean = np.array(Image.open(speaker_file)).mean(axis=1)
    img_std = np.array(Image.open(speaker_file)).std(axis=1)
    img = np.concatenate([img_mean, img_std])
    speaker_vectors.append(img)

speaker_vectors = np.array(speaker_vectors)
speaker_vectors = reduction_method(n_components=2).fit_transform(speaker_vectors)

plt.figure(figsize=(10, 10))
sns.scatterplot(
    x=speaker_vectors[:, 0],
    y=speaker_vectors[:, 1],
    hue=speaker_names,
    legend=False,
)
plt.savefig("figures/speaker_images.png")

speaker_vectors = []
speaker_names = []

for speaker_file in Path("data").glob("**/*.npy"):
    if "_dvec.npy" not in speaker_file.name:
        continue
    if plot_books:
        speaker_name = speaker_file.parent.name
    else:
        speaker_name = speaker_file.parent.parent.name
    speaker_names.append(speaker_name)
    speaker_vectors.append(np.load(speaker_file))

speaker_vectors = np.array(speaker_vectors)
speaker_vectors = reduction_method(n_components=2).fit_transform(speaker_vectors)


plt.figure(figsize=(10, 10))
sns.scatterplot(
    x=speaker_vectors[:, 0],
    y=speaker_vectors[:, 1],
    hue=speaker_names,
    legend=False,
)

plt.savefig("figures/speaker_dvecs.png")
