import numpy as np
def Rx(x):
    res = np.eye(4)
    res[1, 1] = np.cos(x)
    res[1, 2] = -np.sin(x)
    res[2, 1] = np.sin(x)
    res[2, 2] = np.cos(x)
    return res

def Ry(x):
    res = np.eye(4)
    res[0, 0] = np.cos(x)
    res[0, 2] = np.sin(x)
    res[2, 0] = -np.sin(x)
    res[2, 2] = np.cos(x)
    return res

def Rz(x):
    res = np.eye(4)
    res[0, 0] = np.cos(x)
    res[0, 1] = -np.sin(x)
    res[1, 0] = np.sin(x)
    res[1, 1] = np.cos(x)
    return res

def Tx(p):
    res = np.eye(4)
    res[0, 3] = p
    return res

def Ty(p):
    res = np.eye(4)
    res[1, 3] = p
    return res

def Tz(p):
    res = np.eye(4)
    res[2, 3] = p
    return res

class Dh():
    def __init__(self, dh_list, type='mdh', ordermap=['d', 'alpha', 'a']):
        self.dh_list = dh_list
        self.order_map = ordermap
        self.alpha_idx = ordermap.index('alpha')
        self.a_idx = ordermap.index('a')
        self.d_idx = ordermap.index('d')
        self.type = type

    def millimeter_to_meter(self):
        for i in range(len(self.dh_list)):
            self.dh_list[i][self.a_idx] /= 1000
            self.dh_list[i][self.d_idx] /= 1000

    def order_map(self):
        print(self.ordermap)

def mdh_to_matrix_list(dh_params: Dh):
    ''' M = Tx(a)*Rx(alpha)*Tz(d)*Rz(theta)'''
    Ms = []
    for dh in dh_params.dh_list:
        trans = Tx(dh[dh_params.a_idx]) @ Rx(dh[dh_params.alpha_idx]) \
                @ Tz(dh[dh_params.d_idx])
        Ms.append(trans)
    return Ms

def sdh_to_matrix_list(dh_params):
    ''' M = Rz(theta)*Tz(d)*Rx(alpha)*Tx(a)'''
    Ms = []
    for dh in dh_params.dh_list:
        trans = Tz(dh[dh_params.d_idx]) @ Rx(dh[dh_params.alpha_idx]) \
                @ Tx(dh[dh_params.a_idx])
        Ms.append(trans)
    return Ms

def get_matrix_list(dh_params, base, ee):
    '''
    解析dh参数转为连杆间转换矩阵
    如果是sdh,则会多输出一个法兰偏移,这是sdh的特质决定的
    此偏移后续会和ee结合起来
    '''
    res = [base]
    if dh_params.type == 'mdh':
        Ms = mdh_to_matrix_list(dh_params)
        res.extend(Ms)
        res.append(ee)
    elif dh_params.type == 'sdh':
        Ms = sdh_to_matrix_list(dh_params)
        res.append(np.eye(4))
        last_link_trans = Ms.pop()
        res.extend(Ms)
        #todo: sdh的情况下，重新set_ee会出问题
        res.append(last_link_trans @ ee)
    else:
        raise ValueError('dh type error')

    return res
