import re
import numpy as np
from datetime import datetime, timedelta


def extract_data(data_path):
    timestamp_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})'

    datas = [[], [], [], []]
    ts = [[], [], [], []]

    with open(data_path, 'r') as file:
        type = 0
        for line in file:
            line_strip = line.strip()
            time_match = re.match(timestamp_pattern, line_strip)
            if not time_match:
                raise ValueError("timestamp not found")
            group = time_match.group(1)
            timestamp, millisecond = group.split('.')

            base_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            absolute_utc = base_time + \
                timedelta(milliseconds=float(millisecond))

            message = line.split(']')[1].replace(" ", "")
            key = message.split('=')[0]
            values = message.split('=')[1].split(',')
            ts[type].append(absolute_utc.timestamp())
            datas[type].append([float(x) for x in values])

            type = 0 if type == 3 else type + 1

    # * remove large timestamp absolute value
    t_init = ts[0][0]
    t_qc = np.array(ts[0]) - t_init
    t_qa = np.array(ts[1]) - t_init
    t_pc = np.array(ts[2]) - t_init
    t_pa = np.array(ts[3]) - t_init

    t_diff = np.diff(t_qa)
    t_threshold = 8e-3  # * choose this var according to actual control frequency
    continous_mask = t_diff < t_threshold
    t_diff = t_diff[continous_mask]
    avg_dt = np.mean(t_diff) * 1e3
    # avg_dt = (t_qa[-1] - t_qa[0]) / t_qc.shape[0] * 1e3

    save_path = 'npz/cmd.npz'
    np.savez(save_path, q_cmd=datas[0], q_act=datas[1],
             p_cmd=datas[2], p_act=datas[3],
             t_qc=t_qc, t_qa=t_qa, t_pc=t_pc, t_pa=t_pa)
    print(f'[extract] total data_num = {len(ts[0])}')
    print(f'[extract] exctrat data finished, save to {save_path}')
    print(f'[info] avg dt = {avg_dt:.3f} ms')


if __name__ == '__main__':
    data_path = 'data/update_pos.log'
    extract_data(data_path)
