import urllib.request


def download_hemolysis():
    urllib.request.urlretrieve(
        'https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/human-positive.npz',
        './data/positive.npz',
    )
    urllib.request.urlretrieve(
        'https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/human-negative.npz',
        './data/negative.npz',
    )


download_hemolysis()
