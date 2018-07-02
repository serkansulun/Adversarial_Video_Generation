import sys
import os
import torch
from copy import deepcopy

from g_model import GeneratorModel
from d_model import DiscriminatorModel
import constants as c
from batch import Batcher as Batcher
from utils import save_samples, calculate_psnr, calculate_sharp_psnr, RunningAverage
from pytorch_msssim import ssim as calculate_ssim
from loss_functions import combined_loss, adv_loss


class Runner:
    def __init__(self):

        self.g_model = GeneratorModel()
        if c.CUDA:
            self.g_model.cuda()

        self.batcher_tst = Batcher('test')

        if not c.TEST_ONLY:
            self.batcher_trn = Batcher('train')
            self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=c.LRATE_G)

            if c.ADVERSARIAL:
                self.d_model = DiscriminatorModel()
                self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=c.LRATE_D, weight_decay=c.REG_D)

                if c.CUDA:
                    self.d_model.cuda()

            self.epoch = 0

        if c.LOAD_MODEL is not None:
            self.load_model()

    def one_epoch(self, batcher):

        mode = batcher.mode
        if mode == 'train':
            self.g_model.train()
            if c.ADVERSARIAL:
                self.d_model.eval()

        elif mode == 'test':
            self.g_model.eval()
        else:
            sys.exit('Mode can be train or test')

        dataset_done = False
        batcher.new()
        performance = {}
        averager = RunningAverage()

        while not dataset_done:
            # create minibatches
            history, gt, dataset_done = batcher.get()

            # DISCRIMINATOR
            if c.ADVERSARIAL and mode == 'train':
                self.d_model.zero_grad()

                # run generator to create fake samples
                # detach generator from discriminator training
                fake_disc_generation = [generation.detach() for generation in self.g_model(history)]
                # create fake and real sequences
                disc_input_for_disc = []
                for i in range(c.NUM_SCALE_NETS):
                    fake_sequence = torch.cat([history[i], fake_disc_generation[i]], 1)
                    real_sequence = torch.cat([history[i], gt[i]], 1)
                    disc_input_for_disc.append(torch.cat([fake_sequence, real_sequence], 0))

                disc_labels = torch.cat([torch.zeros((fake_sequence.shape[0], 1)),
                                         torch.ones((real_sequence.shape[0], 1))])
                disc_labels.requires_grad = False
                # Run discriminator
                preds = self.d_model(disc_input_for_disc)
                disc_loss = adv_loss(preds, disc_labels)

                # Train discriminator
                disc_loss.backward()
                self.d_optimizer.step()

                performance.update({'disc_loss': disc_loss})
            else:
                preds = None

            # GENERATOR

            self.g_model.zero_grad()

            # Run generator and discriminator again, since generator was previously detached
            generation = self.g_model(history)

            psnr = calculate_psnr(generation[-1], gt[-1])     # calculate PSNR on largest scale
            sharp_psnr = calculate_sharp_psnr(generation[-1], gt[-1])
            ssim = calculate_ssim(generation[-1], gt[-1], val_range=2)

            performance.update({'psnr': psnr, 'sharp_psnr': sharp_psnr, 'ssim': ssim})

            if mode == 'train':
                if c.ADVERSARIAL:
                    disc_input_for_gen = []
                    for i in range(c.NUM_SCALE_NETS):
                        disc_input_for_gen.append(torch.cat([history[i], generation[i]], 1))
                    gen_preds = self.d_model(disc_input_for_gen)
                    gen_labels = torch.ones(gen_preds[0].shape[0], 1)
                    gen_labels.requires_grad = False
                else:
                    gen_preds = None

                gen_loss = combined_loss(generation, gt, preds=gen_preds)
                gen_loss.backward()
                self.g_optimizer.step()

                performance.update({'gen_loss': gen_loss})

        averager.update(performance)
        save_samples(history, gt, generation, self.epoch, mode)

        return averager.values()

    def train(self):
        """
        Runs a training loop on the model networks.
        """
        while self.epoch < c.EPOCHS:
            performance_trn = self.one_epoch(self.batcher_trn)    # train
            performance_tst = self.one_epoch(self.batcher_tst)   # test
            print('Epoch {:2d}: Train: Gen loss: {:.3f}, Disc loss: {:.3f}, Test: PSNR: {:.3f}, Sharp PSNR: {:.3f}, SSIM: {:.3f}'.format(
                  self.epoch+1, performance_trn['gen_loss'], performance_trn['disc_loss'], performance_tst['psnr'],
                  performance_tst['sharp_psnr'], performance_tst['ssim']))

            if c.SAVE_MODEL:
                self.save_model()
            self.epoch += 1

    def test(self):

        performance_tst = self.one_epoch(self.batcher_tst)
        print('PSNR: {:.3f}, Sharp PSNR: {:.3f}, SSIM: {:.3f}'.format(performance_tst['psnr'],
              performance_tst['sharp_psnr'], performance_tst['ssim']))

    def save_model(self):
        g_model_copy = deepcopy(self.g_model)
        data = {'epoch': self.epoch+1, 'gen_optim': self.g_optimizer.state_dict(),
                'gen_model': g_model_copy.cpu().state_dict()}

        if c.ADVERSARIAL:
            d_model_copy = deepcopy(self.d_model)
            data.update({'disc_model': d_model_copy.cpu().state_dict()})
            data.update({'disc_optim': self.d_optimizer.state_dict()})
        torch.save(data, c.MODEL_DIR + '/' + c.EXPERIMENT_NAME + '.pt')

    def load_model(self):

        data = torch.load(os.path.join(c.MODEL_DIR, c.LOAD_MODEL + '.pt'), map_location=lambda storage, loc: storage)
        self.epoch = data['epoch']
        self.g_model.load_state_dict(data['gen_model'])
        self.g_optimizer.load_state_dict(data['gen_optim'])

        # Adjust learning rate to a newly defined one
        for param in self.g_optimizer.param_groups:
            param['lr'] = c.LRATE_G
        if c.ADVERSARIAL and 'disc_model' in data.keys():
            self.d_model.load_state_dict(data['disc_model'])
            self.d_optimizer.load_state_dict(data['disc_optim'])
            for param in self.d_optimizer.param_groups:
                param['lr'] = c.LRATE_D
        print 'Restored model'


def main():

    runner = Runner()
    if c.TEST_ONLY:
        runner.test()
    else:
        runner.train()

if __name__ == '__main__':
    main()
