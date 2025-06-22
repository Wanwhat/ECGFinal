import mat73
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.muti_attention_model import model
from utils import set_seed, resample, baseline_remove, min_max_normalize

if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_data_path = './CPSC2025/testdata.mat'
    test_data = mat73.loadmat(test_data_path)['testdata']

    model = model.to(device)
    model.load_state_dict(torch.load('./checkpoints/best-model-test.pt', map_location=device))

    label_mapping = {
        0: "N",
        1: "AF"
    }

    model.eval()
    with torch.no_grad():
        predict_labels = []
        for i in tqdm(range(len(test_data))):
            data = test_data[i]

            data_resampled = resample(data, sample_rate=400, resample_rate=300)
            data_baseline = baseline_remove(data_resampled, 300)
            data_norm = min_max_normalize(data_baseline)
            data_expand = data_norm[np.newaxis, np.newaxis, :]

            data_torch = torch.from_numpy(data_expand).float().to(device)
            output = model(data_torch)
            predict_label = output.argmax(dim=1).item()
            predict_labels.append(predict_label)

    print(pd.value_counts(predict_labels))
    str_labels = np.array([label_mapping[label] for label in predict_labels])

    file_names = [f"{i + 1}" for i in range(len(test_data))]
    df = pd.DataFrame({
        'file': file_names,
        'label': str_labels
    })

    csv_path = './results.csv'
    df.to_csv(csv_path, index=False)
