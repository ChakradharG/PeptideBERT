import urllib.request


def download_hemolysis():
    urllib.request.urlretrieve(
        'https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/hemo-positive.npz',
        './data/hemo-positive.npz',
    )
    urllib.request.urlretrieve(
        'https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/hemo-negative.npz',
        './data/hemo-negative.npz',
    )


def download_solubility():
    urllib.request.urlretrieve(
        'https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/soluble.npz',
        './data/sol-positive.npz',
    )
    urllib.request.urlretrieve(
        'https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/insoluble.npz',
        './data/sol-negative.npz',
    )


def download_nonfouling():
    urllib.request.urlretrieve(
        'https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/human-positive.npz',
        './data/nf-positive.npz',
    )
    urllib.request.urlretrieve(
        'https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/human-negative.npz',
        './data/nf-negative.npz',
    )


download_hemolysis()
download_solubility()
download_nonfouling()
