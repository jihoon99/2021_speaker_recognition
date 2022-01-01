import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import LabelEncoder


def make_left_right(df, iteration=8):
    '''
    result : dataframe : left, right, label

        |     |  left  |  right  |  label  |
        |  1  |  135   |  156    |    0    |
        |  2  |  11    |  33     |    1    |
                   ...

    result에 있는 left : 135의 뜻은
    밑에 df의 135번째 row를 뜻합니다.
    하지만 저 df의 mfcc는 source dict에 저장하였습니다.
    source[135]를 하면 left 135의 mfcc가 뽑혀나옵니다.
    -> result가 split 되어서 dataset에 source와 같이 들어갑니다.


    df : filename, speaker, pick

        |      |  filename    | speaker  | pick  |
        |   1  | root/001.wav | idx0001  |   1   |
        |   2  | root/002.wav | idx0001  |   1   | 
                 ...

    pick 뜻은 몇번 뽑혓나 볼려고- 공평하게 장가 시집 보내려고

    df는 60,000개의  row를 갖고 있음.


    '''
    df['pick'] = 0
    # df has [file_name, speaker, pick]

    result = []  # 쌍 {'left':left, "right":right, 'label': 1 or 0}
    np.random.seed(42)

    speaker_order = df.groupby("speaker").count().sort_values("pick").reset_index().speaker

    for _ in tqdm.tqdm(range(iteration)):
        # df = df.sample(frac=1).reset_index(drop=True)
        # speaker_order = df.speaker.unique()
        for sp in speaker_order:  # 0 ~ 10
            in_tmp = df[df.speaker == sp]
            in_in = in_tmp.sample(frac=0.55, replace=False) # 0.66이 0.4: 0.6정도 비율이였음.

            # 0.66 국내로, 0.33 외국으로 가는거지
            # 짝수로 맞추기
            if len(in_in) % 2 != 0:
                in_in = in_in[:-1]

            # 국내로
            in_in_idx = list(in_in.index)
            in_out_idx = [idx for idx in list(
                in_tmp.index) if idx not in in_in_idx]

            np.random.shuffle(in_in_idx)
            np.random.shuffle(in_out_idx)

            df.pick.loc[in_in_idx] = df.pick.loc[in_in_idx] + 1
            df.pick.loc[in_out_idx] = df.pick.loc[in_out_idx] + 1
            for left, right in zip(in_in_idx[::2], in_in_idx[1::2]):
                result.append(
                    {'left_path': left, 'right_path': right, 'label': 1})

            # 외국으로
            out_tmp = df[df.speaker != sp]
            # globaly pick 본거지... 외국에서 가장 인기 없는 친구한테 배정
            out_min = min(out_tmp.pick)
            out_min_idx_1 = list(out_tmp.loc[out_tmp.pick == out_min].index)

            if len(out_min_idx_1) < len(in_out_idx):
                out_min_idx_2 = list(
                    out_tmp.loc[out_tmp.pick == out_min+1].index)
                np.random.shuffle(out_min_idx_2)

                out_chosen_idx = out_min_idx_1 + \
                    out_min_idx_2[:(len(in_out_idx)-len(out_min_idx_1))]

            else:
                np.random.shuffle(out_min_idx_1)
                out_chosen_idx = out_min_idx_1[:len(in_out_idx)]

            for left, right in zip(in_out_idx, out_chosen_idx):
                result.append(
                    {'left_path': left, 'right_path': right, 'label': 0})
                df.pick.iloc[right] = df.pick.iloc[right] + 1

    result = pd.DataFrame(result).drop_duplicates()

    print(f'label 1 : {sum(result.label == 1)/len(result)}')
    print(f'label 0 : {sum(result.label == 0)/len(result)}')

    return result, df
