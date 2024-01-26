import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CN2 Adult
data = {
    "train_time": [7.08072186, 33.70691395, 52.02528882, 126.72904086, 76.47666097, 71.17876482,
                   116.47588015, 86.27651095, 67.64891386, 64.97003818, 84.55489707, 126.93838525,
                   107.62231898, 89.53514385, 59.43629599, 65.32682085, 68.83326411, 88.46064591,
                   69.08030295],
    "test_time": [0.00348115, 0.00321507, 0.00501919, 0.00495124, 0.00485826, 0.00547504, 0.00588012,
                  0.00476384, 0.00475097, 0.00554991, 0.00619006, 0.00678301, 0.00563312, 0.00490808,
                  0.004812, 0.00502396, 0.00572276, 0.00534511, 0.00470614],
    "accuracy": [1.0, 0.77400216, 0.77015908, 0.78960195, 0.75906736, 0.75799884, 0.77468201, 0.80568012,
                 0.77161215, 0.74343208, 0.75275013, 0.76488706, 0.76398467, 0.79877301, 0.79109589, 0.76695157,
                 0.75657895, 0.75395034, 0.77414075],
    "recall": [1.0, 0.52138493, 0.61471861, 0.54545455, 0.62833676, 0.59429825, 0.55952381, 0.60544218,
               0.57407407, 0.54791155, 0.49518304, 0.54990215, 0.60830861, 0.51675978, 0.54208754, 0.5430622,
               0.49234136, 0.57597173, 0.53281853],
    "F1": [1.0, 0.54994629, 0.57548126, 0.57331137, 0.56824513, 0.565762, 0.5026738, 0.57792208, 0.55918828,
           0.49281768, 0.52129817, 0.55098039, 0.57103064, 0.53008596, 0.51355662, 0.52607184, 0.5033557,
           0.4992343, 0.5],
    "no_rules": [255, 252, 442, 507, 347, 488, 510, 437, 357, 450, 511, 538, 548, 389, 414, 382, 466, 493, 837]
}

# CN2 Bank
cn2_bank = {
    "train_time": [0.73694015, 1.18117094, 0.90421605, 1.0274601, 2.00026011,
                   2.31019998, 3.40449977, 2.85287619, 1.97160196, 2.73206711,
                   2.2293179, 1.60004997, 1.14906287, 2.29755998, 2.5950458,
                   1.95956874, 2.40676475, 1.60905719, 2.14621902, 2.28580594],
    "test_time": [0.0012579, 0.00131083, 0.00116491, 0.00116873, 0.00142598,
                  0.00158906, 0.00135612, 0.00146294, 0.00136614, 0.0013392,
                  0.0015018, 0.0012989, 0.00139403, 0.00193, 0.00150919,
                  0.0013001, 0.00135016, 0.00121808, 0.00133896, 0.001297],
    "accuracy": [1.0, 0.75438596, 0.94230769, 0.74537037, 0.8671875,
                 0.90822785, 0.5546875, 0.46753247, 0.875, 0.54761905,
                 0.5515873, 0.9, 0.66470588, 0.50442478, 0.73584906,
                 0.7826087, 0.7591623, 0.79651163, 0.48905109, 0.62937063],
    "recall": [1.0, 0.0, 0.0, 0.25, 0.11764706,
               0.125, 0.01754386, 0.33333333, 0.06666667, 0.1,
               0.2, 0.09090909, 0.05660377, 0.39285714, 0.5,
               0.1025641, 0.11111111, 0.02857143, 0.35294118, 0.34375],
    "F1": [1.0, 0.0, 0.0, 0.20289855, 0.10526316,
           0.06451613, 0.03389831, 0.03755869, 0.08163265, 0.17391304,
           0.0173913, 0.10526316, 0.0952381, 0.28205128, 0.125,
           0.18604651, 0.04166667, 0.05405405, 0.25531915, 0.29333333],
    "no_rules": [42, 42, 22, 36, 59,
                 63, 73, 64, 47, 71,
                 51, 41, 36, 79, 58,
                 55, 60, 38, 70, 54]
}

df = pd.DataFrame(data)

# Function to plot and save each metric
FIGURES_FOLDER = os.path.abspath(os.path.join(__file__, "..", "..", "..", "reports", "figures", "stage-2", "CN2"))


def plot_and_save(metric_name):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x=df.index, y=metric_name, color='#AEE0D5', linewidth=2.5)
    plt.title(metric_name.replace('_', ' ').title())
    plt.xlabel('')
    plt.grid(True)
    sns.despine()
    plt.savefig(f'{FIGURES_FOLDER}/adult-{metric_name}.eps', format='eps')
    plt.close()


# Plotting and saving each metric
plot_and_save('train_time')
plot_and_save('test_time')
plot_and_save('accuracy')
plot_and_save('recall')
plot_and_save('F1')
