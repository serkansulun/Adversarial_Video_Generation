import time
import datetime
import constants as c
import cPickle as pickle


class Performance:
    def __init__(self):
        self.t1 = time.time()
        self.datetime = datetime.datetime.now().strftime("%m-%d--%H-%M")
        self.date = datetime.datetime.now().strftime("%m-%d")
        self.total_counter = 0
        self.epoch_begin = True
        self.program_begin = True
        self.misc = {'steps': [], 'samples': [], 'time': [], 'tst_psnr': [], 'tst_sharpdiff': []}

    def update(self, performance):

        if self.program_begin:
            self.performance_array = performance.copy()
            for key in performance.keys():
                self.performance_array[key] = []
            self.program_begin = False

        if self.epoch_begin:
            self.average = performance.copy()
            self.counter = 0
            for key in performance.keys():
                self.average[key] = 0

            self.epoch_begin = False

        # take moving average
        for key in performance.keys():
            self.average[key] = (self.average[key] * self.counter + performance[key]) / (self.counter + 1)
        self.counter += 1
        self.total_counter += 1

    def save(self, dataset_size, tst_psnr_av=None, tst_sharpdiff_av=None):
        t2 = time.time()
        # and save performance
        for key in self.average.keys():
            self.performance_array[key].append(self.average[key])

        self.misc['tst_psnr'].append(tst_psnr_av)
        self.misc['tst_sharpdiff'].append(tst_sharpdiff_av)
        self.misc['steps'].append(self.total_counter)
        self.misc['samples'].append(self.total_counter * c.BATCH_SIZE)
        self.misc['time'].append((t2 - self.t1) / 60.0)

        self.constants = {'dataset_size': dataset_size, 'date': self.date, 'datetime': self.datetime,
                          'lam_adv': c.LAM_ADV, 'lam_gdl': c.LAM_GDL, 'lam_lp': c.LAM_LP,
                          'gen_lr': c.LRATE_G, 'disc_lr': c.LRATE_D}

        f_p = open(c.PERFORMANCE_SAVE_DIR + '/' + c.EXPERIMENT + '_' + self.date + '.p', "wb")
        data = dict(self.performance_array.items() + self.misc.items() + self.constants.items())

        pickle.dump(data, f_p)
        f_p.close()

        # reset averages
        self.epoch_begin = True
